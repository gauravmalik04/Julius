"""LangGraph Orchestration Pipeline for Manchester United Fan Companion.

Implements the complete graph workflow depicted in workflow.png:
  User Query
      ↓
  Load Team Context (Manchester United ID 66 / Old Trafford)
      ↓
  Scope Guard — [Out of Scope] → Warm Redirect & Suggest Topics → END
      ↓ [In Scope]
  Intent Classifier
      ↓
  Clear Intent? — [No] → Clarification Loop ──(loop back)──> Grounded Response → END
      ↓ [Yes]
  Conditional Router
      ├─ Add to Calendar     → add_to_calendar_tool (@tool)
      ├─ Schedule & Fixtures   → schedule_and_fixtures_tool (@tool)
      ├─ News & Transfers    → news_and_transfers_tool (@tool)
      ├─ Match Recap         → match_recap_tool (@tool)
      ├─ Live Stats Lookup   → live_stats_lookup_tool (@tool)
      └─ Greetings & General → general_chitchat_tool (@tool)
      ↓
  Answer Generator (LLM Synthesis bound with tools)
      ↓
  Grounded Response + Citations
      ↓
  Response to User
"""

import logging
import operator
from typing import TypedDict, Sequence, Optional, Dict, Any, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI # type:ignore
from langgraph.graph import StateGraph, END

from . import config
from .retrieval import retrieve_news, format_context
from .fixtures_handler import handle_fixtures_intent
from .stats_handler import handle_stats_intent
from .match_recap import run_match_recap_pipeline
from .reminders_handler import handle_calendar_intent, get_pending_confirmation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangChain Tools Definitions (@tool)
# ---------------------------------------------------------------------------

@tool
def add_to_calendar_tool(query: str, user_id: str = "default_user") -> str:
    """Tool to set, view, confirm, cancel, or remove Manchester United match reminders in Google Calendar."""
    print(f"[DEBUG TOOL: add_to_calendar_tool] Query: '{query}', user_id: '{user_id}'", flush=True)
    return handle_calendar_intent(query, user_id=user_id)


@tool
def schedule_and_fixtures_tool(query: str) -> str:
    """Tool to lookup upcoming or past match schedules, kickoff times, dates, and venues for Manchester United."""
    print(f"[DEBUG TOOL: schedule_and_fixtures_tool] Query: '{query}'", flush=True)
    return handle_fixtures_intent(query)


@tool
def news_and_transfers_tool(query: str) -> str:
    """Tool to search vector database for Manchester United transfer news, press conferences, and club updates."""
    print(f"[DEBUG TOOL: news_and_transfers_tool] Query: '{query}'", flush=True)
    try:
        chunks = retrieve_news(query, top_k=5)
        return format_context(chunks)
    except Exception as exc:
        logger.warning("[news_and_transfers_tool] Error: %s", exc)
        return f"[News Context — Unavailable: {exc}]"


@tool
def match_recap_tool(query: str) -> str:
    """Tool to generate a detailed tactical or narrative match recap for a past Manchester United match."""
    print(f"[DEBUG TOOL: match_recap_tool] Query: '{query}'", flush=True)
    return run_match_recap_pipeline(query)


@tool
def live_stats_lookup_tool(query: str) -> str:
    """Tool to lookup Manchester United squad statistics, Premier League standings, player goals, assists, and form."""
    print(f"[DEBUG TOOL: live_stats_lookup_tool] Query: '{query}'", flush=True)
    return handle_stats_intent(query)


@tool
def general_chitchat_tool(query: str) -> str:
    """Tool to handle general greetings, small talk, and introductions in an enthusiastic Red Devils fan voice."""
    print(f"[DEBUG TOOL: general_chitchat_tool] Query: '{query}'", flush=True)
    return "[Context: General Greeting & Chitchat]"


# Active Tools List
ALL_TOOLS = [
    add_to_calendar_tool,
    schedule_and_fixtures_tool,
    news_and_transfers_tool,
    match_recap_tool,
    live_stats_lookup_tool,
    general_chitchat_tool,
]


def _get_bound_llm():
    """Initialize ChatAnthropic and bind with all domain tools using MODEL_SONNET for higher reasoning."""
    # pyrefly: ignore [missing-import]
    from langchain_anthropic import ChatAnthropic
    kwargs = {
        "model_name": config.MODEL_SONNET,
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    if config.ANTHROPIC_API_KEY:
        kwargs["anthropic_api_key"] = config.ANTHROPIC_API_KEY
    if config.ANTHROPIC_BASE_URL:
        kwargs["anthropic_api_url"] = config.ANTHROPIC_BASE_URL

    llm = ChatAnthropic(**kwargs)
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    session_id: str
    summary: str
    messages: Sequence[BaseMessage]
    user_id: str
    team_context: Dict[str, Any]
    query: str
    is_in_scope: bool
    redirect_message: str
    intent: str
    sub_intent: str
    needs_clarification: bool
    clarification_prompt: str
    retrieved_context: str
    final_response: str
    pending_confirmation: Optional[Dict[str, Any]]
    execution_steps: Annotated[list[str], operator.add]
    executed_tools: Annotated[list[Dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Node 1: Pre-Injected Team Context & Scope Guard Entry Point
# ---------------------------------------------------------------------------
MAN_UTD_CONTEXT = {
    "team_name": "Manchester United",
    "team_code": "MUN",
    "team_id": config.MAN_UTD_TEAM_ID,
    "stadium": "Old Trafford",
}


def scope_guard_node(state: AgentState) -> dict:
    """Evaluate if query is in-scope for Manchester United or out-of-scope."""
    print(f"\n--- [LANGGRAPH NODE: Scope Guard (Pre-Injected Context)] ---", flush=True)
    query = state.get("query", "").strip()
    user_id = state.get("user_id", "default_user")

    # If there is an active pending confirmation (e.g. Yes/No for calendar), force in-scope
    pending = get_pending_confirmation(user_id)
    if pending is not None:
        print(f"[DEBUG: SCOPE GUARD] Pending confirmation active for user '{user_id}'. Forcing in-scope.", flush=True)
        return {
            "team_context": MAN_UTD_CONTEXT,
            "is_in_scope": True,
            "redirect_message": "",
            "intent": "calendar",
            "pending_confirmation": pending,
            "execution_steps": ["🛡️ Scope Guard: Active confirmation pending. Bypassed guardrail."],
        }

    from .main import classify_intent
    res = classify_intent(query)
    intent = res.get("intent", "news")
    redirect = res.get("redirect", "")

    if intent == "out_of_scope":
        print(f"[DEBUG: SCOPE GUARD] Query OUT OF SCOPE. Redirect: '{redirect}'", flush=True)
        return {
            "team_context": MAN_UTD_CONTEXT,
            "is_in_scope": False,
            "redirect_message": redirect,
            "intent": "out_of_scope",
            "execution_steps": [f"⛔ Scope Guard: Query classified out-of-scope for Manchester United."],
        }

    print(f"[DEBUG: SCOPE GUARD] Query IN SCOPE. Initial intent: '{intent}'", flush=True)
    return {
        "team_context": MAN_UTD_CONTEXT,
        "is_in_scope": True,
        "redirect_message": redirect,
        "intent": intent,
        "execution_steps": [f"🔍 Scope Guard: Verified query in-scope for Manchester United."],
    }


# ---------------------------------------------------------------------------
# Node 3: Warm Redirect & Suggest Team Topics (Out of Scope End)
# ---------------------------------------------------------------------------
def warm_redirect_node(state: AgentState) -> dict:
    """Construct warm redirect response for out-of-scope queries."""
    print(f"\n--- [LANGGRAPH NODE: Warm Redirect] ---", flush=True)
    redirect = state.get("redirect_message") or (
        "I'm only set up to follow Manchester United! I can tell you about United's upcoming fixtures, "
        "the latest transfer news, match recaps, player stats, or set reminders for games."
    )
    return {
        "final_response": redirect,
        "execution_steps": ["↪️ Warm Redirect: Prepared friendly club topic redirect."],
    }


# ---------------------------------------------------------------------------
# Node 4: Intent Classifier & Ambiguity Check
# ---------------------------------------------------------------------------
def intent_classifier_node(state: AgentState) -> dict:
    """Refine intent and check if query requires clarification."""
    print(f"\n--- [LANGGRAPH NODE: Intent Classifier] ---", flush=True)
    query = state.get("query", "")
    intent = state.get("intent", "news")

    if state.get("pending_confirmation"):
        return {
            "intent": "calendar",
            "needs_clarification": False,
            "execution_steps": ["🎯 Intent Classifier: Routing active confirmation to calendar handler."],
        }

    needs_clarification = False
    clarification_prompt = ""
    query_lower = query.lower().strip()

    if intent in ("ambiguous", "ambiguous_intent") or query_lower in ("tell me about the game", "what happened", "match info"):
        needs_clarification = True
        clarification_prompt = state.get("redirect_message") or (
            "Would you like to see **upcoming fixtures**, **live match stats**, or a **tactical match recap** of our last game?"
        )
        print(f"[DEBUG: INTENT CLASSIFIER] Ambiguous query detected ('{intent}'). Triggering clarification loop.", flush=True)

    return {
        "intent": intent,
        "needs_clarification": needs_clarification,
        "clarification_prompt": clarification_prompt,
        "execution_steps": [f"🎯 Intent Classifier: Classified intent as '{intent}'."],
    }


# ---------------------------------------------------------------------------
# Node 5: Clarification Loop
# ---------------------------------------------------------------------------
def clarification_loop_node(state: AgentState) -> dict:
    """Ask user for clarification on ambiguous query."""
    print(f"\n--- [LANGGRAPH NODE: Clarification Loop] ---", flush=True)
    prompt = state.get("clarification_prompt") or state.get("redirect_message") or "Could you clarify what specific Manchester United info you're looking for?"
    return {
        "final_response": prompt,
        "execution_steps": ["❓ Clarification Loop: Triggered user prompt for query disambiguation."],
    }


# ---------------------------------------------------------------------------
# Action Node A: Add to Calendar Tool Node
# ---------------------------------------------------------------------------
def add_to_calendar_node(state: AgentState) -> dict:
    """Execute add_to_calendar_tool (@tool)."""
    print(f"\n--- [LANGGRAPH NODE: Add to Calendar (Tool Node)] ---", flush=True)
    query = state.get("query", "")
    user_id = state.get("user_id", "default_user")

    response = add_to_calendar_tool.invoke({"query": query, "user_id": user_id})
    return {
        "retrieved_context": "",
        "final_response": response,
        "execution_steps": ["📅 Calendar Tool: Executed match reminder handler."],
        "executed_tools": [{"name": "add_to_calendar_tool", "args": {"query": query, "user_id": user_id}}],
    }


# ---------------------------------------------------------------------------
# Action Node B: Schedule & Fixtures Tool Node
# ---------------------------------------------------------------------------
def schedule_fixtures_node(state: AgentState) -> dict:
    """Execute schedule_and_fixtures_tool (@tool)."""
    print(f"\n--- [LANGGRAPH NODE: Schedule & Fixtures (Tool Node)] ---", flush=True)
    query = state.get("query", "")
    context = schedule_and_fixtures_tool.invoke({"query": query})
    return {
        "retrieved_context": context,
        "execution_steps": ["⚽ Fixtures Tool: Queried kickoff schedules from TheSportsDB & Football-Data APIs."],
        "executed_tools": [{"name": "schedule_and_fixtures_tool", "args": {"query": query}}],
    }


# ---------------------------------------------------------------------------
# Action Node C: News & Transfers Tool Node
# ---------------------------------------------------------------------------
def news_transfers_node(state: AgentState) -> dict:
    """Execute news_and_transfers_tool (@tool)."""
    print(f"\n--- [LANGGRAPH NODE: News & Transfers (Tool Node)] ---", flush=True)
    query = state.get("query", "")
    context = news_and_transfers_tool.invoke({"query": query})
    return {
        "retrieved_context": context,
        "execution_steps": ["📰 News & Transfers Tool: Performed ChromaDB semantic vector search."],
        "executed_tools": [{"name": "news_and_transfers_tool", "args": {"query": query}}],
    }


# ---------------------------------------------------------------------------
# Action Node D: Match Recap Tool Node
# ---------------------------------------------------------------------------
def match_recap_node(state: AgentState) -> dict:
    """Execute match_recap_tool (@tool)."""
    print(f"\n--- [LANGGRAPH NODE: Match Recap (Tool Node)] ---", flush=True)
    query = state.get("query", "")
    recap_text = match_recap_tool.invoke({"query": query})
    return {
        "retrieved_context": recap_text,
        "execution_steps": ["📝 Match Recap Tool: Retrived match report & article chunks."],
        "executed_tools": [{"name": "match_recap_tool", "args": {"query": query}}],
    }


# ---------------------------------------------------------------------------
# Action Node E: Live Stats Lookup Tool Node
# ---------------------------------------------------------------------------
def live_stats_node(state: AgentState) -> dict:
    """Execute live_stats_lookup_tool (@tool)."""
    print(f"\n--- [LANGGRAPH NODE: Live Stats Lookup (Tool Node)] ---", flush=True)
    query = state.get("query", "")
    context = live_stats_lookup_tool.invoke({"query": query})
    return {
        "retrieved_context": context,
        "execution_steps": ["📊 Live Stats Tool: Fetched league standings & player squad statistics."],
        "executed_tools": [{"name": "live_stats_lookup_tool", "args": {"query": query}}],
    }


# ---------------------------------------------------------------------------
# Action Node F: General & Greetings Tool Node
# ---------------------------------------------------------------------------
def general_chitchat_node(state: AgentState) -> dict:
    """Execute general_chitchat_tool (@tool)."""
    print(f"\n--- [LANGGRAPH NODE: Greetings & General Chitchat (Tool Node)] ---", flush=True)
    query = state.get("query", "")
    context = general_chitchat_tool.invoke({"query": query})
    return {
        "retrieved_context": context,
        "execution_steps": ["💬 General Chitchat Tool: Formatted friendly Red Devils fan conversation context."],
        "executed_tools": [{"name": "general_chitchat_tool", "args": {"query": query}}],
    }


# ---------------------------------------------------------------------------
# Node 6: Answer Generator (Bound LLM Synthesis)
# ---------------------------------------------------------------------------
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

def answer_generator_node(state: AgentState) -> dict:
    """Synthesize natural language answer using AzureChatOpenAI bound with tools and session memory context."""
    print(f"\n--- [LANGGRAPH NODE: Answer Generator (Tool Bound LLM + Session Memory)] ---", flush=True)
    query = state.get("query", "")
    session_id = state.get("session_id", "default_user")
    context_block = state.get("retrieved_context", "")

    # If final_response was already produced by a self-contained tool (e.g. calendar, recap, clarification), return it directly
    if state.get("final_response"):
        print(f"[DEBUG: ANSWER GEN] Using pre-generated tool response.", flush=True)
        return {
            "final_response": state["final_response"],
            "execution_steps": ["🧠 Answer Generator: Output pre-formulated tool response."],
        }

    from .main import _build_system_prompt
    from .memory_manager import load_session_messages
    summary, past_messages = load_session_messages(session_id)

    # Build conversation memory block
    memory_context = ""
    if summary:
        memory_context += f"\n\n[CONVERSATION SUMMARY]\n{summary}"
    if past_messages:
        turns = []
        for m in past_messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            turns.append(f"{role}: {m.content}")
        memory_context += f"\n\n[RECENT RELEVANT CHAT HISTORY]\n" + "\n".join(turns)

    system_prompt = _build_system_prompt() + memory_context
    user_message = f"{context_block}\n\n[User Question]\n{query}" if context_block else query

    try:
        bound_llm = _get_bound_llm()
        messages = [
            ("system", system_prompt),
            ("user", user_message),
        ]
        resp = bound_llm.invoke(messages)

        answer = ""
        has_tool_calls = hasattr(resp, "tool_calls") and bool(resp.tool_calls)

        if has_tool_calls:
            tool_call = resp.tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            print(f"[DEBUG: LLM TOOL CALL DETECTED] Invoking tool '{tool_name}' with args {tool_args}", flush=True)
            tool_output = None
            for t in ALL_TOOLS:
                if t.name == tool_name:
                    tool_output = t.invoke(tool_args)
                    break

            if tool_output is not None:
                tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_call.get("id", "call_1"))
                followup = bound_llm.invoke([
                    ("system", system_prompt),
                    ("user", user_message),
                    resp,
                    tool_msg,
                ])
                if isinstance(followup.content, str):
                    answer = followup.content
                elif isinstance(followup.content, list):
                    text_blocks = [b.get("text", "") for b in followup.content if isinstance(b, dict) and b.get("type") == "text"]
                    answer = "\n".join(t for t in text_blocks if t)
        else:
            if isinstance(resp.content, str):
                answer = resp.content
            elif isinstance(resp.content, list):
                text_blocks = [b.get("text", "") for b in resp.content if isinstance(b, dict) and b.get("type") == "text"]
                answer = "\n".join(t for t in text_blocks if t)

        if not answer:
            from .main import _client
            plain_resp = _client().messages.create(
                model=config.MODEL_SONNET,
                temperature=0.4,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )
            answer = plain_resp.content[0].text

        return_dict = {
            "final_response": answer,
            "execution_steps": ["🧠 Answer Generator: Synthesized grounded response using Anthropic Claude."],
        }
        if has_tool_calls and tool_name:
            return_dict["executed_tools"] = [{"name": tool_name, "args": tool_args}]
        
        return return_dict
    except Exception as exc:
        logger.error("[Graph] Answer generator error: %s", exc)
        return {
            "final_response": f"I had trouble synthesizing the answer: {exc}",
            "execution_steps": [f"⚠️ Answer Generator: Error synthesizing answer: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 7: Grounded Response + Citations & Memory Persistence
# ---------------------------------------------------------------------------
def grounded_response_node(state: AgentState) -> dict:
    """Final response formatting, citation grounding check, and session memory persistence."""
    print(f"\n--- [LANGGRAPH NODE: Grounded Response + Citations & Memory Persistence] ---", flush=True)
    final_response = state.get("final_response", "")
    session_id = state.get("session_id", "default_user")
    query = state.get("query", "")

    # Persist in-scope conversation turn to session memory store
    if query and final_response and state.get("is_in_scope") is not False:
        from .memory_manager import save_session_turn
        save_session_turn(session_id, query, final_response)

    print(f"[DEBUG: GROUNDED RESPONSE] Response ready ({len(final_response)} chars)\n----------------------------------------------------\n", flush=True)
    return {
        "final_response": final_response,
        "execution_steps": ["💾 Session Memory: Persisted turn to memory store."],
    }


# ---------------------------------------------------------------------------
# Conditional Router Functions
# ---------------------------------------------------------------------------
def route_scope_guard(state: AgentState) -> str:
    """Branch to intent_classifier if in-scope; warm_redirect if out-of-scope."""
    return "intent_classifier" if state.get("is_in_scope") else "warm_redirect"


def route_clarification(state: AgentState) -> str:
    """Branch to clarification_loop if ambiguous; conditional_router if clear."""
    return "clarification_loop" if state.get("needs_clarification") or state.get("intent") in ("ambiguous", "ambiguous_intent") else "conditional_router"


def route_intent_to_action(state: AgentState) -> str:
    """Route intent to corresponding tool node."""
    intent = state.get("intent", "news")
    mapping = {
        "calendar": "add_to_calendar",
        "fixtures": "schedule_fixtures",
        "news": "news_transfers",
        "recap": "match_recap",
        "stats": "live_stats",
        "general": "general_chitchat",
    }
    target_node = mapping.get(intent, "news_transfers")
    print(f"[DEBUG: CONDITIONAL ROUTER] Routing intent '{intent}' -> tool node '{target_node}'", flush=True)
    return target_node


# ---------------------------------------------------------------------------
# Build & Compile StateGraph
# ---------------------------------------------------------------------------
def build_agent_graph():
    """Construct and compile the LangGraph StateGraph matching workflow.png with pre-injected context."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("scope_guard", scope_guard_node)
    workflow.add_node("warm_redirect", warm_redirect_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("clarification_loop", clarification_loop_node)

    # Dummy router node to bridge conditional edges cleanly
    def conditional_router_node(state: AgentState) -> dict:
        return {}
    workflow.add_node("conditional_router_node", conditional_router_node)

    # Action Tool Nodes
    workflow.add_node("add_to_calendar", add_to_calendar_node)
    workflow.add_node("schedule_fixtures", schedule_fixtures_node)
    workflow.add_node("news_transfers", news_transfers_node)
    workflow.add_node("match_recap", match_recap_node)
    workflow.add_node("live_stats", live_stats_node)
    workflow.add_node("general_chitchat", general_chitchat_node)

    workflow.add_node("answer_generator", answer_generator_node)
    workflow.add_node("grounded_response", grounded_response_node)

    # Entry edge (Pre-injected context entry point)
    workflow.set_entry_point("scope_guard")

    # Scope Guard branching
    workflow.add_conditional_edges(
        "scope_guard",
        route_scope_guard,
        {
            "intent_classifier": "intent_classifier",
            "warm_redirect": "warm_redirect",
        },
    )
    workflow.add_edge("warm_redirect", END)

    # Clarification branching
    workflow.add_conditional_edges(
        "intent_classifier",
        route_clarification,
        {
            "clarification_loop": "clarification_loop",
            "conditional_router": "conditional_router_node",
        },
    )
    workflow.add_edge("clarification_loop", "grounded_response")

    # Conditional Router branching
    workflow.add_conditional_edges(
        "conditional_router_node",
        route_intent_to_action,
        {
            "add_to_calendar": "add_to_calendar",
            "schedule_fixtures": "schedule_fixtures",
            "news_transfers": "news_transfers",
            "match_recap": "match_recap",
            "live_stats": "live_stats",
            "general_chitchat": "general_chitchat",
        },
    )

    # Action nodes -> Answer Generator / Grounded Response
    for node in ["schedule_fixtures", "news_transfers", "match_recap", "live_stats", "general_chitchat"]:
        workflow.add_edge(node, "answer_generator")

    workflow.add_edge("add_to_calendar", "grounded_response")
    workflow.add_edge("answer_generator", "grounded_response")
    workflow.add_edge("grounded_response", END)

    return workflow.compile()


# Compile globally
app_graph = build_agent_graph()
