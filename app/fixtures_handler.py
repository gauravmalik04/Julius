"""Fixtures handler for the Manchester United Fan Companion.

Flow:
  handle_fixtures_intent(message)
        ↓
  extract_fixture_intent()   — LLM mini-call → structured sub-intent
        ↓
  fetch_team_matches()       — football-data.org /v4/teams/66/matches
        ↓
  filter_matches()           — client-side home/away + opponent filter
        ↓
  format_fixture_context()   — clean [Fixture Context] block with UTC + IST times
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAN_UTD_ID = config.MAN_UTD_TEAM_ID  # 66 on football-data.org
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Opponent name → football-data.org team ID lookup map
# Covers all Premier League clubs + common European opponents Man Utd face.
# ---------------------------------------------------------------------------
_TEAM_ALIASES: dict[str, int] = {
    # Premier League
    "arsenal": 57, "gunners": 57,
    "chelsea": 61, "the blues": 61,
    "liverpool": 64, "the reds": 64,
    "manchester city": 65, "man city": 65, "city": 65,
    "tottenham": 73, "spurs": 73, "tottenham hotspur": 73, "hotspur": 73,
    "newcastle": 67, "newcastle united": 67, "newcastle utd": 67, "magpies": 67,
    "aston villa": 58, "villa": 58,
    "west ham": 563, "west ham united": 563, "hammers": 563,
    "brighton": 397, "brighton & hove albion": 397, "seagulls": 397,
    "everton": 62, "toffees": 62,
    "brentford": 402, "bees": 402,
    "fulham": 63, "cottagers": 63,
    "crystal palace": 354, "palace": 354, "eagles": 354,
    "wolves": 76, "wolverhampton": 76, "wolverhampton wanderers": 76,
    "nottingham forest": 351, "forest": 351, "nffc": 351,
    "bournemouth": 1044, "afc bournemouth": 1044, "cherries": 1044,
    "leicester": 338, "leicester city": 338, "foxes": 338,
    "ipswich": 349, "ipswich town": 349,
    "southampton": 340, "saints": 340,
    "luton": 1031, "luton town": 1031, "hatters": 1031,
    "burnley": 328, "clarets": 328,
    "sheffield united": 356, "sheffield utd": 356, "blades": 356,
    "sunderland": 69,
    "leeds": 341, "leeds united": 341,
    "watford": 346,
    # European opponents (Champions League / Europa League)
    "juventus": 109, "juve": 109,
    "real madrid": 86, "madrid": 86,
    "barcelona": 81, "barca": 81,
    "atletico madrid": 78, "atletico": 78,
    "bayern": 5, "bayern munich": 5, "fc bayern": 5,
    "psg": 524, "paris saint-germain": 524, "paris sg": 524,
    "inter milan": 108, "inter": 108, "internazionale": 108,
    "ac milan": 98, "milan": 98,
    "roma": 100, "as roma": 100,
    "lazio": 110,
    "napoli": 113,
    "porto": 4, "fc porto": 4,
    "benfica": 294, "sl benfica": 294,
    "sporting": 228, "sporting cp": 228,
    "ajax": 674,
    "rangers": 371,
    "celtic": 1,
    "galatasaray": 442,
    "fenerbahce": 440,
    "bayer leverkusen": 3, "leverkusen": 3,
    "borussia dortmund": 16, "dortmund": 16, "bvb": 16,
    "rb leipzig": 721, "leipzig": 721,
    "sevilla": 559,
    "villarreal": 585,
    "young boys": 1871,
}


def _resolve_opponent_id(name: str) -> Optional[int]:
    """Fuzzy lookup: lowercase + strip → match in alias map."""
    return _TEAM_ALIASES.get(name.lower().strip())


# ---------------------------------------------------------------------------
# Step 1 — Extract structured fixture sub-intent from user message
# ---------------------------------------------------------------------------
_FIXTURE_INTENT_PROMPT = """\
You are an intent extractor for a Manchester United chatbot. The user has asked a fixtures-related question.

Your job is to return a JSON object with exactly two fields:

1. "sub_intent" — one of:
   - "next_match"              : next upcoming match (any venue)
   - "next_home"               : next upcoming home match at Old Trafford (no specific opponent)
   - "next_away"               : next upcoming away match (no specific opponent)
   - "last_match"              : most recent past match (any venue)
   - "last_home"               : most recent past home match (no specific opponent)
   - "last_away"               : most recent past away match (no specific opponent)
   - "vs_opponent_next"        : next upcoming match against a specific opponent (venue not specified)
   - "vs_opponent_last"        : most recent past match against a specific opponent (venue not specified)
   - "vs_opponent_home_next"   : next upcoming HOME match at Old Trafford against a specific opponent
   - "vs_opponent_home_last"   : most recent past HOME match at Old Trafford against a specific opponent
   - "vs_opponent_away_next"   : next upcoming AWAY match against a specific opponent
   - "vs_opponent_away_last"   : most recent past AWAY match against a specific opponent

2. "opponent_name" — the opponent's name as mentioned by the user, or "" if not applicable.

Rules:
- If the user mentions both a specific team AND "home" or "at Old Trafford", use vs_opponent_home_*.
- If the user mentions both a specific team AND "away", use vs_opponent_away_*.
- If the user mentions a specific team with no venue context, use vs_opponent_next or vs_opponent_last.
- "past", "last", "previous", "result", "how did we do" → prefer last_* sub_intents.
- "next", "upcoming", "when do we play", "fixture" → prefer next_* sub_intents.
- If time direction is ambiguous (e.g. "vs Arsenal" with no context), use "vs_opponent_next".
- Respond ONLY with valid JSON. No markdown, no explanation.

Examples:
  User: "When's our next game?"                          → {"sub_intent": "next_match", "opponent_name": ""}
  User: "Next home match?"                               → {"sub_intent": "next_home", "opponent_name": ""}
  User: "When's the next away fixture?"                  → {"sub_intent": "next_away", "opponent_name": ""}
  User: "What was our last match?"                       → {"sub_intent": "last_match", "opponent_name": ""}
  User: "Last home result?"                              → {"sub_intent": "last_home", "opponent_name": ""}
  User: "Last away game?"                                → {"sub_intent": "last_away", "opponent_name": ""}
  User: "When do we play Arsenal?"                       → {"sub_intent": "vs_opponent_next", "opponent_name": "Arsenal"}
  User: "How did we do against Chelsea?"                 → {"sub_intent": "vs_opponent_last", "opponent_name": "Chelsea"}
  User: "When's the next home game against Arsenal?"     → {"sub_intent": "vs_opponent_home_next", "opponent_name": "Arsenal"}
  User: "Last home match vs Man City?"                   → {"sub_intent": "vs_opponent_home_last", "opponent_name": "Man City"}
  User: "Next away game against Liverpool?"              → {"sub_intent": "vs_opponent_away_next", "opponent_name": "Liverpool"}
  User: "Last time we played away at Spurs?"             → {"sub_intent": "vs_opponent_away_last", "opponent_name": "Spurs"}
"""


def extract_fixture_intent(message: str) -> dict:
    """
    Use a small LLM call to extract structured fixture sub-intent from the user's message.

    Returns:
      {
        "sub_intent":    one of the 8 values above,
        "opponent_name": "Arsenal" | "" (only for vs_opponent_* sub-intents)
      }
    Falls back to {"sub_intent": "next_match", "opponent_name": ""} on any error.
    """
    import json as _json
    from openai import AzureOpenAI # type:ignore

    try:
        client = AzureOpenAI(
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
        )
        resp = client.chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": _FIXTURE_INTENT_PROMPT},
                {"role": "user", "content": message},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        result = _json.loads(raw)
        sub_intent = result.get("sub_intent", "next_match")
        opponent_name = result.get("opponent_name", "")
        logger.info("Fixture sub-intent: %s | opponent: '%s'", sub_intent, opponent_name)
        print(f"[DEBUG: FIXTURE INTENT] sub_intent='{sub_intent}' opponent='{opponent_name}'")
        return {"sub_intent": sub_intent, "opponent_name": opponent_name}
    except Exception as exc:
        logger.warning("Fixture intent extraction failed (%s) — defaulting to next_match", exc)
        return {"sub_intent": "next_match", "opponent_name": ""}


# ---------------------------------------------------------------------------
# Step 2 — Fetch matches from football-data.org
# ---------------------------------------------------------------------------
def _fetch_fd_matches(future: bool) -> list[dict]:
    """Fetch Man Utd matches from football-data.org using date-range filtering."""
    api_key = os.environ.get("FOOTBALL_API_KEY") or config.FOOTBALL_API_KEY
    if not api_key:
        logger.warning("FOOTBALL_API_KEY is not set.")
        return []

    today = datetime.now(timezone.utc).date()
    if future:
        date_from = today.isoformat()
        date_to = (today + timedelta(days=180)).isoformat()
        label = "UPCOMING"
    else:
        date_from = (today - timedelta(days=180)).isoformat()
        date_to = (today - timedelta(days=1)).isoformat()
        label = "PAST"

    url = f"{config.FOOTBALL_API_BASE}/teams/{MAN_UTD_ID}/matches"
    params = {"dateFrom": date_from, "dateTo": date_to}
    headers = {"X-Auth-Token": api_key}

    print(f"[DEBUG: FIXTURES API] GET {url}?dateFrom={date_from}&dateTo={date_to} ({label})", flush=True)

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        print(f"[DEBUG: FIXTURES API] Received {len(matches)} {label} matches from football-data.org.", flush=True)
        return matches
    except Exception as exc:
        logger.error("football-data.org error: %s", exc)
        return []


def _fetch_sdb_matches(future: bool) -> list[dict]:
    """Fetch from TheSportsDB and map to football-data.org schema."""
    api_key = (config.SPORTS_DB_API_KEY or "").strip() or "3"
    base_url = f"{config.SPORTS_DB_BASE}/{api_key}"
    
    try:
        resp = requests.get(f"{base_url}/searchteams.php", params={"t": "Manchester United"}, timeout=10)
        resp.raise_for_status()
        teams = resp.json().get("teams")
        team_id = teams[0]["idTeam"] if teams else "133612"
    except Exception:
        team_id = "133612"

    endpoint = "eventsnext.php" if future else "eventslast.php"
    label = "UPCOMING" if future else "PAST"
    print(f"[DEBUG: FIXTURES API] GET {base_url}/{endpoint}?id={team_id} ({label})", flush=True)

    try:
        resp = requests.get(f"{base_url}/{endpoint}", params={"id": team_id}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events") or data.get("results") or []
        print(f"[DEBUG: FIXTURES API] Received {len(events)} {label} matches from TheSportsDB.", flush=True)
    except Exception as exc:
        logger.error("TheSportsDB error: %s", exc)
        return []

    mapped_matches = []
    for e in events:
        date_str = e.get("dateEvent", "")
        time_str = e.get("strTime", "00:00:00")
        try:
            utc_date = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            utc_date = (e.get("strTimestamp") or "").replace("Z", "+00:00")
            
        status = "SCHEDULED" if future else "FINISHED"
        if e.get("strStatus") in ("Match Finished", "FT"):
            status = "FINISHED"

        home_id = e.get("idHomeTeam")
        away_id = e.get("idAwayTeam")
        # Ensure our MAN_UTD_ID matches Man Utd so that client-side filters work
        home_id = MAN_UTD_ID if home_id == team_id else (home_id or 0)
        away_id = MAN_UTD_ID if away_id == team_id else (away_id or 0)
        
        home_score = e.get("intHomeScore")
        away_score = e.get("intAwayScore")
        try:
            home_score = int(home_score) if home_score is not None else None
            away_score = int(away_score) if away_score is not None else None
        except Exception:
            home_score = away_score = None
            
        winner = None
        if home_score is not None and away_score is not None:
            if home_score > away_score: winner = "HOME_TEAM"
            elif away_score > home_score: winner = "AWAY_TEAM"
            else: winner = "DRAW"

        mapped_matches.append({
            "utcDate": utc_date,
            "status": status,
            "matchday": e.get("intRound"),
            "competition": {"name": e.get("strLeague", "Unknown Competition")},
            "homeTeam": {"id": home_id, "name": e.get("strHomeTeam", "Unknown")},
            "awayTeam": {"id": away_id, "name": e.get("strAwayTeam", "Unknown")},
            "venue": e.get("strVenue", ""),
            "score": {
                "fullTime": {"home": home_score, "away": away_score},
                "winner": winner
            },
            "_source": "TheSportsDB"
        })
        
    return mapped_matches


def fetch_team_matches(future: bool) -> list[dict]:
    """
    Fetch matches from both TheSportsDB and football-data.org.
    Merges both lists and deduplicates by match date and opponent name,
    prioritizing TheSportsDB.
    """
    sdb_matches = _fetch_sdb_matches(future)
    fd_matches = _fetch_fd_matches(future)
    
    merged = []
    seen_keys = set()
    
    # Priority 1: TheSportsDB
    for m in sdb_matches:
        date_str = m["utcDate"][:10] if m["utcDate"] else ""
        home_name = m["homeTeam"]["name"]
        away_name = m["awayTeam"]["name"]
        opponent = away_name if "manchester" in home_name.lower() or "man utd" in home_name.lower() else home_name
        key = f"{date_str}_{opponent.lower()}"
        seen_keys.add(key)
        merged.append(m)
        
    # Priority 2: football-data.org (add only if not already present)
    for m in fd_matches:
        date_str = m["utcDate"][:10] if m["utcDate"] else ""
        home_name = m["homeTeam"]["name"]
        away_name = m["awayTeam"]["name"]
        opponent = away_name if "manchester" in home_name.lower() or "man utd" in home_name.lower() else home_name
        key = f"{date_str}_{opponent.lower()}"
        
        if key not in seen_keys:
            merged.append(m)
            
    if not merged:
        raise RuntimeError("No fixtures found from both APIs.")
        
    return merged


def _filter_by_opponent(matches: list[dict], opponent_name: str) -> list[dict]:
    """
    Return matches where the named opponent appears in either team slot.
    Tries ID lookup first; falls back to case-insensitive name substring match.
    """
    opponent_id = _resolve_opponent_id(opponent_name)
    if opponent_id:
        return [
            m for m in matches
            if (m.get("homeTeam") or {}).get("id") == opponent_id
            or (m.get("awayTeam") or {}).get("id") == opponent_id
        ]
    # Fallback: text-match on team names (case-insensitive)
    opp_lower = opponent_name.lower().strip()
    return [
        m for m in matches
        if opp_lower in (m.get("homeTeam") or {}).get("name", "").lower()
        or opp_lower in (m.get("awayTeam") or {}).get("name", "").lower()
    ]


# ---------------------------------------------------------------------------
# Step 3 — Client-side filtering
# ---------------------------------------------------------------------------
def filter_matches(
    matches: list[dict],
    sub_intent: str,
    opponent_name: str = "",
) -> list[dict]:
    """
    Filter and sort the match list based on sub_intent and optional opponent_name.

    Home/away:              compare homeTeam.id / awayTeam.id against MAN_UTD_ID
    vs_opponent:            match opponent in either team slot (any venue)
    vs_opponent_home_*:     opponent in awayTeam slot (United are the home team)
    vs_opponent_away_*:     opponent in homeTeam slot (United are the away team)
    last_*:                 sort descending (most recent first)
    next_* / vs_*_next:     API returns ascending (soonest first)
    """
    filtered: list[dict] = []

    if sub_intent in ("next_match", "last_match"):
        filtered = matches[:]

    elif sub_intent in ("next_home", "last_home"):
        filtered = [
            m for m in matches
            if (m.get("homeTeam") or {}).get("id") == MAN_UTD_ID
        ]

    elif sub_intent in ("next_away", "last_away"):
        filtered = [
            m for m in matches
            if (m.get("awayTeam") or {}).get("id") == MAN_UTD_ID
        ]

    elif sub_intent in ("vs_opponent_next", "vs_opponent_last"):
        # Any venue — opponent appears in either team slot
        filtered = _filter_by_opponent(matches, opponent_name)

    elif sub_intent in ("vs_opponent_home_next", "vs_opponent_home_last"):
        # United are at home → opponent must be the away team
        opp_matches = _filter_by_opponent(matches, opponent_name)
        filtered = [
            m for m in opp_matches
            if (m.get("homeTeam") or {}).get("id") == MAN_UTD_ID
        ]

    elif sub_intent in ("vs_opponent_away_next", "vs_opponent_away_last"):
        # United are away → opponent must be the home team
        opp_matches = _filter_by_opponent(matches, opponent_name)
        filtered = [
            m for m in opp_matches
            if (m.get("awayTeam") or {}).get("id") == MAN_UTD_ID
        ]

    # Sort: last_* → descending (most recent first); next_* / vs_*_next → ascending
    reverse = (
        sub_intent.startswith("last_")
        or sub_intent.endswith("_last")
    )
    filtered.sort(key=lambda m: m.get("utcDate", ""), reverse=reverse)

    print(f"[DEBUG: FIXTURES FILTER] {len(filtered)} matches after filtering for '{sub_intent}'")
    return filtered


# ---------------------------------------------------------------------------
# Step 4 — Format match data into an LLM-injectable context block
# ---------------------------------------------------------------------------
def _parse_utc(utc_date_str: str) -> Optional[datetime]:
    """Parse the API's ISO 8601 UTC date string to a timezone-aware datetime."""
    if not utc_date_str:
        return None
    try:
        return datetime.fromisoformat(utc_date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_datetime(utc_date_str: str) -> str:
    """Return a human-readable string with both UTC and IST times."""
    dt = _parse_utc(utc_date_str)
    if not dt:
        return "Date TBC"
    utc_str = dt.strftime("%a, %d %b %Y — %H:%M UTC")
    ist_dt = dt.astimezone(IST)
    ist_str = ist_dt.strftime("%H:%M IST")
    return f"{utc_str} ({ist_str})"


def format_fixture_context(matches: list[dict], sub_intent: str) -> str:
    """
    Format up to 3 match dicts into a clean [Fixture Context] block
    ready to be injected into the LLM prompt.

    Includes: competition, matchday, date (UTC + IST), teams, score (if finished), venue.
    """
    if not matches:
        return ""

    heading_map = {
        "next_match":            "Next Match",
        "next_home":             "Next Home Match",
        "next_away":             "Next Away Match",
        "last_match":            "Last Match",
        "last_home":             "Last Home Match",
        "last_away":             "Last Away Match",
        "vs_opponent_next":      "Next Match vs Opponent",
        "vs_opponent_last":      "Last Match vs Opponent",
        "vs_opponent_home_next": "Next Home Match vs Opponent",
        "vs_opponent_home_last": "Last Home Match vs Opponent",
        "vs_opponent_away_next": "Next Away Match vs Opponent",
        "vs_opponent_away_last": "Last Away Match vs Opponent",
    }
    heading = heading_map.get(sub_intent, "Fixture")

    lines = [f"[Fixture Context — {heading}]"]

    for i, match in enumerate(matches[:3], 1):
        competition = (match.get("competition") or {}).get("name", "Unknown Competition")
        matchday = match.get("matchday")
        matchday_str = f" (Matchday {matchday})" if matchday else ""

        home_team = (match.get("homeTeam") or {}).get("name", "Unknown")
        away_team = (match.get("awayTeam") or {}).get("name", "Unknown")

        date_str = _format_datetime(match.get("utcDate", ""))
        status = match.get("status", "UNKNOWN")
        venue = match.get("venue", "")

        lines.append(f"\n--- Match {i} ---")
        lines.append(f"Competition : {competition}{matchday_str}")
        lines.append(f"Date        : {date_str}")
        lines.append(f"Home        : {home_team}")
        lines.append(f"Away        : {away_team}")

        if venue:
            lines.append(f"Venue       : {venue}")

        # Score — only present for FINISHED matches
        score_info = match.get("score") or {}
        full_time = score_info.get("fullTime") or {}
        home_score = full_time.get("home")
        away_score = full_time.get("away")

        if status == "FINISHED" and home_score is not None and away_score is not None:
            winner = score_info.get("winner", "")
            if winner == "HOME_TEAM":
                result_note = f"({home_team} won)"
            elif winner == "AWAY_TEAM":
                result_note = f"({away_team} won)"
            else:
                result_note = "(Draw)"
            lines.append(
                f"Score       : {home_team} {home_score} – {away_score} {away_team}  {result_note}"
            )
        else:
            lines.append(f"Status      : {status}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entrypoint — orchestrates all 4 steps
# ---------------------------------------------------------------------------
def handle_fixtures_intent(message: str) -> str:
    """
    Orchestrate the full fixtures lookup pipeline:
      extract_fixture_intent → fetch_team_matches → filter_matches → format_fixture_context

    Returns a formatted [Fixture Context] string for LLM injection,
    or an informative fallback string if no matches are found or an error occurs.
    """
    print(f"\n[DEBUG: FIXTURES HANDLER] Processing: '{message}'", flush=True)

    # Step 1: Classify sub-intent
    intent_result = extract_fixture_intent(message)
    sub_intent = intent_result["sub_intent"]
    opponent_name = intent_result["opponent_name"]

    # Step 2: Determine direction (future vs past)
    is_future = sub_intent in (
        "next_match", "next_home", "next_away",
        "vs_opponent_next",
        "vs_opponent_home_next",
        "vs_opponent_away_next",
    )

    # Step 3: Fetch from football-data.org
    try:
        matches = fetch_team_matches(future=is_future)
    except RuntimeError as exc:
        logger.error("Fixtures API config error: %s", exc)
        return f"[Fixture Context — Unavailable]\n{exc}"
    except requests.HTTPError as exc:
        logger.error("Fixtures API HTTP error: %s", exc)
        return "[Fixture Context — Unavailable]\nThe fixtures service is temporarily unavailable."

    # Step 4: Filter client-side
    filtered = filter_matches(matches, sub_intent, opponent_name)

    if not filtered:
        direction = "upcoming" if is_future else "past"
        if opponent_name:
            return (
                f"[Fixture Context — No Results]\n"
                f"No {direction} matches found against '{opponent_name}' "
                f"in the current season. They may have been from a previous season "
                f"or a competition not in the free data tier."
            )
        return (
            f"[Fixture Context — No Results]\n"
            f"No {direction} matches found for the requested filter. "
            f"The current season data may be complete."
        )

    # Step 5: Format and return
    context = format_fixture_context(filtered, sub_intent)
    print(f"[DEBUG: FIXTURES HANDLER] Context ready ({len(filtered)} match(es))", flush=True)
    return context
