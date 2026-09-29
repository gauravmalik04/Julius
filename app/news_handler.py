"""News & Injury Handler for Manchester United Fan Companion.

Provides sub-intent extraction for 'news' intent:
  - 'injuries'         : fitness, injury news, player availability, sidelined players
  - 'transfers'        : transfer updates, signings, rumors, contract extensions
  - 'press_conference' : manager quotes, pre-match press conferences
  - 'general'          : club updates, general news

Uses API-Football (https://v3.football.api-sports.io) via API_FOOTBALL_KEY for 'injuries' sub-intent,
combined with semantic RAG vector retrieval from ChromaDB news_store.
"""

import json
import logging
import re
import urllib.request
from typing import Optional, Dict, Any, List

from . import config
from .retrieval import retrieve_news, format_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-intent classifier prompt for News Intent
# ---------------------------------------------------------------------------
_NEWS_SUBINTENT_PROMPT = """\
You are an intent parser for a Manchester United fan companion chatbot.
Analyze the user's message regarding news, transfers, or injuries, and return a JSON object with two fields:

1. "sub_intent": one of:
   - "injuries"         : asking about player injuries, fitness, availability, sidelined players, or recovery (e.g. "Is Mount fit?", "Who is injured?")
   - "transfers"        : asking about transfers, signings, rumors, or contract extensions (e.g. "Any transfer news?")
   - "press_conference" : asking about manager press conferences, pre-match statements, or quotes
   - "general"          : general club announcements, news, or unspecified news updates

2. "player_name": string containing the specific player or manager name referenced if applicable, otherwise null.

Examples:
  User: "Is Mason Mount fit for the next game?"
  --> {"sub_intent": "injuries", "player_name": "Mason Mount"}

  User: "Who is currently on the injury list?"
  --> {"sub_intent": "injuries", "player_name": null}

  User: "Are there any updates on Højlund's hamstring injury?"
  --> {"sub_intent": "injuries", "player_name": "Rasmus Højlund"}

  User: "Did United sign any new midfielders?"
  --> {"sub_intent": "transfers", "player_name": null}

  User: "What did the manager say in the press conference?"
  --> {"sub_intent": "press_conference", "player_name": null}
"""


def extract_news_subintent(message: str) -> dict:
    """Parse user query into structured news sub-intent using Anthropic MODEL_HAIKU."""
    from .main import _client
    try:
        resp = _client().messages.create(
            model=config.MODEL_HAIKU,
            temperature=0,
            max_tokens=100,
            system=_NEWS_SUBINTENT_PROMPT,
            messages=[{"role": "user", "content": message}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n|\n```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        sub_intent = result.get("sub_intent", "general")
        player_name = result.get("player_name")
        print(f"[DEBUG: NEWS SUB-INTENT] sub_intent='{sub_intent}', player_name='{player_name}'", flush=True)
        return {"sub_intent": sub_intent, "player_name": player_name}
    except Exception as exc:
        logger.warning("[NewsHandler] Sub-intent extraction failed: %s", exc)
        return {"sub_intent": "general", "player_name": None}


# ---------------------------------------------------------------------------
# API-Football Injuries Integration (https://v3.football.api-sports.io/injuries)
# ---------------------------------------------------------------------------
def fetch_api_football_injuries(season: Optional[int] = None) -> List[dict]:
    """
    Fetch injury data for Manchester United (team ID 33) from API-Football.
    API-Football uses API_FOOTBALL_KEY with header 'x-apisports-key'.
    """
    api_key = config.API_FOOTBALL_KEY
    if not api_key:
        logger.warning("[API-Football] API_FOOTBALL_KEY is missing from environment.")
        return []

    seasons_to_try = [season] if season else [2025, 2024, 2026]
    for s in seasons_to_try:
        url = f"{config.API_FOOTBALL_BASE}/injuries?team={config.MAN_UTD_API_FOOTBALL_ID}&season={s}"
        print(f"[DEBUG: API-FOOTBALL] GET {url}", flush=True)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "x-apisports-key": api_key,
                    "User-Agent": "ManUtdFanCompanion/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                records = data.get("response", [])
                if records:
                    print(f"[DEBUG: API-FOOTBALL] Fetched {len(records)} injury records for season {s}.", flush=True)
                    return records
        except Exception as exc:
            logger.warning("[API-Football] Request error for season %s: %s", s, exc)

    return []


def handle_injuries_intent(query: str, target_player: Optional[str] = None) -> str:
    """
    Fetch and format injury & fitness status using latest RAG news vector search and API-Football structured injury data.
    """
    lines = []

    # 1. Retrieve latest current news articles from RAG vector database (ChromaDB)
    try:
        news_chunks = retrieve_news(query, top_k=4)
        if news_chunks:
            lines.append("[Latest Current News & Injury Updates Context — RAG Vector Database]")
            lines.append(format_context(news_chunks))
    except Exception as exc:
        logger.warning("[NewsHandler] Vector news retrieval error: %s", exc)

    # 2. Fetch and aggregate API-Football structured injury records
    injuries_raw = fetch_api_football_injuries()
    player_injuries: Dict[str, dict] = {}
    for item in injuries_raw:
        p_info = item.get("player", {})
        p_name = p_info.get("name", "Unknown Player")
        p_reason = p_info.get("reason", "Injury")
        p_type = p_info.get("type", "Missing Fixture")
        fixture_date = item.get("fixture", {}).get("date", "")

        # Skip non-injury exclusions (like cards or tactical choices)
        if any(ign in p_reason.lower() for ign in ["red card", "yellow card", "coach's decision", "inactive"]):
            continue

        if p_name not in player_injuries or fixture_date > player_injuries[p_name]["latest_date"]:
            player_injuries[p_name] = {
                "name": p_name,
                "reason": p_reason,
                "type": p_type,
                "latest_date": fixture_date,
            }

    # Sort players by most recent fixture date descending
    sorted_players = sorted(player_injuries.values(), key=lambda x: x["latest_date"], reverse=True)

    lines.append("\n[Official Team Injury Records Context — API-Football (v3.football.api-sports.io)]")
    if sorted_players:
        filtered_players = []
        if target_player:
            t_lower = target_player.lower()
            for p in sorted_players:
                if any(part in p["name"].lower() for part in t_lower.split() if len(part) > 2) or t_lower in p["name"].lower():
                    filtered_players.append(p)

        display_list = filtered_players if filtered_players else sorted_players[:10]
        lines.append(f"Recorded Sidelined Players ({len(display_list)} listed, sorted by recency):")
        for p in display_list:
            date_str = p['latest_date'][:10] if p['latest_date'] else ""
            lines.append(f"- **{p['name']}**: {p['reason']} ({p['type']}{f', as of {date_str}' if date_str else ''})")
    else:
        lines.append("No active injury exclusions currently reported on API-Football.")

    return "\n".join(lines)


def handle_news_intent(query: str) -> str:
    """
    Main handler for the 'news' intent.
    Extracts news sub-intent and routes to API-Football injury handler or RAG vector search.
    """
    print(f"\n--- [NEWS HANDLER] Incoming Query: '{query}' ---", flush=True)
    parsed = extract_news_subintent(query)
    sub_intent = parsed.get("sub_intent", "general")
    player_name = parsed.get("player_name")

    if sub_intent == "injuries":
        return handle_injuries_intent(query, target_player=player_name)

    # For transfers, press_conference, general news: perform semantic RAG retrieval from ChromaDB
    chunks = retrieve_news(query, top_k=5)
    return format_context(chunks)
