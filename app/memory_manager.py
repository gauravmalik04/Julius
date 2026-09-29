"""Conversation Memory Manager for Manchester United Fan Companion.

Implements a Hybrid Sliding Window + Summary Buffer Memory Strategy:
- Stores conversation turns per session_id in data/chat_history.json.
- Keeps the most recent 6 messages (3 turn pairs) verbatim for immediate context.
- When history exceeds 10 messages, older messages are summarized via Azure OpenAI into a running summary.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from . import config

logger = logging.getLogger(__name__)

HISTORY_FILE = config.DATA_DIR / "chat_history.json"
MAX_RECENT_MESSAGES = 6     # Keep 6 recent messages (3 turn pairs)
SUMMARIZE_THRESHOLD = 10    # Summarize when total messages exceed 10


def _ensure_history_file() -> None:
    """Ensure data/chat_history.json exists."""
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("{}", encoding="utf-8")


def load_session_data(session_id: str = "default_user") -> Dict[str, Any]:
    """
    Load raw session data from chat_history.json.
    Returns dict with keys: 'summary' (str), 'messages' (list of dicts [{'role': ..., 'content': ...}])
    """
    _ensure_history_file()
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        session = data.get(session_id, {})
        return {
            "summary": session.get("summary", ""),
            "messages": session.get("messages", []),
        }
    except Exception as exc:
        logger.warning("[MemoryManager] Failed to load session data for '%s': %s", session_id, exc)
        return {"summary": "", "messages": []}


def load_session_messages(session_id: str = "default_user") -> Tuple[str, List[BaseMessage]]:
    """
    Load session history as LangChain BaseMessage objects.
    Returns (running_summary: str, message_objects: List[BaseMessage]).
    """
    raw = load_session_data(session_id)
    summary = raw.get("summary", "")
    message_objs: List[BaseMessage] = []
    for msg in raw.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            message_objs.append(HumanMessage(content=content))
        elif role == "assistant":
            message_objs.append(AIMessage(content=content))
    return summary, message_objs


def summarize_older_messages(old_messages: List[Dict[str, str]], existing_summary: str) -> str:
    """
    Call Anthropic MODEL_SONNET to summarize older conversation turns into a running summary.
    """
    from .main import _client
    print(f"\n[DEBUG MEMORY] Summarizing {len(old_messages)} older messages with Anthropic {config.MODEL_SONNET}...", flush=True)

    formatted_turns = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in old_messages])
    prompt = (
        "You are an assistant summarizing a conversation between a Manchester United fan and their AI fan companion.\n"
        "Compress the provided past conversation turns into a concise, bulleted running summary capturing key facts, "
        "player stats discussed, fixture queries, or reminders set.\n\n"
        f"EXISTING SUMMARY:\n{existing_summary if existing_summary else 'None'}\n\n"
        f"NEW TURNS TO SUMMARIZE:\n{formatted_turns}\n\n"
        "UPDATED CONCISE RUNNING SUMMARY:"
    )

    try:
        resp = _client().messages.create(
            model=config.MODEL_SONNET,
            temperature=0.3,
            max_tokens=300,
            system="You are a concise conversation summarizer.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        new_summary = resp.content[0].text.strip()
        print(f"[DEBUG MEMORY] Summary updated:\n{new_summary}\n", flush=True)
        return new_summary
    except Exception as exc:
        logger.error("[MemoryManager] Summarization failed: %s", exc)
        return existing_summary


def save_session_turn(session_id: str, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
    """
    Append a new conversation turn (user + assistant), handle sliding window & summarization, and persist to chat_history.json.
    """
    _ensure_history_file()
    try:
        full_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        full_data = {}

    session = full_data.get(session_id, {"summary": "", "messages": []})
    summary = session.get("summary", "")
    messages = session.get("messages", [])

    # Append new turn
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "content": assistant_msg})

    # Check summarization threshold
    if len(messages) > SUMMARIZE_THRESHOLD:
        print(f"[DEBUG MEMORY] Messages count ({len(messages)}) exceeded threshold ({SUMMARIZE_THRESHOLD}). Trimming & summarizing...", flush=True)
        # Split into older messages to summarize and recent messages to keep
        num_to_trim = len(messages) - MAX_RECENT_MESSAGES
        old_to_summarize = messages[:num_to_trim]
        recent_to_keep = messages[num_to_trim:]

        summary = summarize_older_messages(old_to_summarize, summary)
        messages = recent_to_keep

    session["summary"] = summary
    session["messages"] = messages
    full_data[session_id] = session

    HISTORY_FILE.write_text(json.dumps(full_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DEBUG MEMORY] Session '{session_id}' saved. Messages count: {len(messages)}, Summary length: {len(summary)}", flush=True)
    return session


def clear_session_memory(session_id: str = "default_user") -> bool:
    """Clear conversation history for a session."""
    _ensure_history_file()
    try:
        full_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if session_id in full_data:
            del full_data[session_id]
            HISTORY_FILE.write_text(json.dumps(full_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[DEBUG MEMORY] Session '{session_id}' history cleared.", flush=True)
        return True
    except Exception as exc:
        logger.error("[MemoryManager] Failed to clear session '%s': %s", session_id, exc)
        return False
