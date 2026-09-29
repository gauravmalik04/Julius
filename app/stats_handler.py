"""Stats handler for the Manchester United Fan Companion.

Flow:
  handle_stats_intent(message)
        ↓
  extract_stats_subintent()   — LLM mini-call → structured sub-intent
        ↓
  validate_squad_member()     — Check local records.json squad & manager
        ↓
  get_standings_context() / get_match_stats_context() / get_player_stats_context()
        ↓
  format context block for main LLM injection
"""

import json
import logging
import re
from functools import lru_cache
from typing import Optional

import requests
from openai import AzureOpenAI

from . import config

logger = logging.getLogger(__name__)

MAN_UTD_ID = config.MAN_UTD_TEAM_ID  # 66 on football-data.org


# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------
def _client():
    """Return an Anthropic client using ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL."""
    import anthropic
    kwargs = {}
    if config.ANTHROPIC_API_KEY:
        kwargs["api_key"] = config.ANTHROPIC_API_KEY
    if config.ANTHROPIC_BASE_URL:
        kwargs["base_url"] = config.ANTHROPIC_BASE_URL
    return anthropic.Anthropic(**kwargs)


# ---------------------------------------------------------------------------
# Step 1 — Squad & Manager Local Data & Syncing
# ---------------------------------------------------------------------------
def sync_squad_data() -> dict:
    """
    Fetch Manchester United team data (squad + coach/manager) from Football-Data.org (/v4/teams/66)
    and update data/records.json with 'squad' and 'manager' records.
    Called on startup / demand or via weekly Sunday scheduler job.
    """
    api_key = (config.FOOTBALL_API_KEY or "").strip()
    headers = {"X-Auth-Token": api_key} if api_key else {}
    url = f"{config.FOOTBALL_API_BASE}/teams/{MAN_UTD_ID}"

    logger.info("[StatsSync] Fetching squad data from Football-Data.org...")
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        squad_list = data.get("squad", [])
        coach = data.get("coach", {})

        records_file = config.DATA_DIR / "records.json"
        existing_records = []
        if records_file.exists():
            try:
                existing_records = json.loads(records_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("[StatsSync] Error reading records.json: %s", e)

        fixtures = [r for r in existing_records if r.get("type") == "fixture" or "type" not in r]
        
        # Check for existing manager in records.json to avoid overwriting custom/updated manager data
        existing_mgr = next((r for r in existing_records if r.get("type") == "manager"), None)
        
        if coach and coach.get("name"):
            mgr_record = {
                "type": "manager",
                "id": f"COACH-{coach.get('id', 0)}",
                "name": coach.get("name"),
                "date_of_birth": coach.get("dateOfBirth"),
                "nationality": coach.get("nationality"),
                "contract": coach.get("contract") or {},
            }
        elif existing_mgr and existing_mgr.get("name"):
            mgr_record = existing_mgr
        else:
            mgr_record = {
                "type": "manager",
                "id": "COACH-0",
                "name": "Michael Carrick",
                "date_of_birth": "1981-07-28",
                "nationality": "England",
                "contract": {},
            }

        squad_records = []
        for player in squad_list:
            squad_records.append({
                "type": "squad",
                "id": f"PLY-{player.get('id')}",
                "name": player.get("name"),
                "position": player.get("position"),
                "date_of_birth": player.get("dateOfBirth"),
                "nationality": player.get("nationality"),
            })

        final_records = fixtures + [mgr_record] + squad_records
        records_file.write_text(json.dumps(final_records, indent=2), encoding="utf-8")
        logger.info("[StatsSync] Saved %d players and coach to records.json", len(squad_list))
        return {
            "status": "ok",
            "squad_count": len(squad_list),
            "manager": coach.get("name", "Unknown"),
        }
    except Exception as exc:
        logger.error("[StatsSync] Failed to sync squad data: %s", exc)
        return {"status": "error", "detail": str(exc)}


def load_squad_and_manager() -> tuple[list[dict], Optional[dict]]:
    """Load squad and manager entries from records.json."""
    records_file = config.DATA_DIR / "records.json"
    if not records_file.exists():
        return [], None

    try:
        data = json.loads(records_file.read_text(encoding="utf-8"))
        squad = [r for r in data if r.get("type") == "squad"]
        manager = next((r for r in data if r.get("type") == "manager"), None)
        return squad, manager
    except Exception as e:
        logger.error("[StatsHandler] Failed to read records.json: %s", e)
        return [], None


def validate_squad_member(query_name: str) -> tuple[bool, Optional[dict], str]:
    """
    Check if a player or manager name exists in the local squad/manager records.

    Returns:
        (is_valid, entity_record, entity_type) where entity_type is 'player', 'manager', or 'unknown'
    """
    if not query_name or len(query_name.strip()) < 2:
        return False, None, "unknown"

    clean_query = query_name.lower().strip()
    squad, manager = load_squad_and_manager()

    # Sync automatically if squad is empty
    if not squad and not manager:
        sync_squad_data()
        squad, manager = load_squad_and_manager()

    # 1. Check manager
    if manager:
        mgr_name = manager.get("name", "").lower()
        if clean_query in ("manager", "coach", "head coach") or clean_query in mgr_name or any(part in clean_query for part in mgr_name.split() if len(part) > 3):
            return True, manager, "manager"

    # Common nickname/alias mapping for Man Utd players
    aliases = {
        "bruno": "Bruno Fernandes",
        "rashford": "Marcus Rashford",
        "hojlund": "Rasmus Højlund",
        "höjlund": "Rasmus Højlund",
        "garnacho": "Alejandro Garnacho",
        "mainoo": "Kobbie Mainoo",
        "casemiro": "Casemiro",
        "onana": "André Onana",
        "dalot": "Diogo Dalot",
        "shaw": "Luke Shaw",
        "maguire": "Harry Maguire",
        "martinez": "Lisandro Martínez",
        "martínez": "Lisandro Martínez",
        "diallo": "Amad Diallo",
        "amad": "Amad Diallo",
        "antony": "Antony",
        "zirkzee": "Joshua Zirkzee",
        "ugarte": "Manuel Ugarte",
        "yoro": "Leny Yoro",
        "de ligt": "Matthijs de Ligt",
        "mazraoui": "Noussair Mazraoui",
        "eriksen": "Christian Eriksen",
        "mount": "Mason Mount",
    }

    target_name = aliases.get(clean_query, clean_query)

    # 2. Check squad
    for player in squad:
        p_name = player.get("name", "").lower()
        p_parts = [part.lower() for part in p_name.split()]

        if target_name in p_parts or target_name == p_name or (len(target_name) >= 4 and target_name in p_name):
            return True, player, "player"

        # Check last name / first name match
        for part in p_parts:
            if len(part) >= 3 and (part in clean_query or clean_query in part):
                return True, player, "player"

    return False, None, "unknown"


# ---------------------------------------------------------------------------
# Step 2 — Sub-Intent Extraction
# ---------------------------------------------------------------------------
_STATS_SUBINTENT_PROMPT = """\
You are an intent parser for a Manchester United fan companion chatbot.
Analyse the user's stats query and return a JSON object with:

1. "sub_intent": one of "player_stats", "manager_stats", "match_stats", "standings", "general"
2. "player_name": player name if asking about a player, else null
3. "manager": true if asking about the manager/coach, else false
4. "timeframe": "last_game" | "season" | "overall" | null
5. "opponent": opponent team name if asking about a specific match (e.g. "Chelsea"), else null

Examples:
- "How many goals did Bruno score in the last game?"
  --> {"sub_intent": "player_stats", "player_name": "Bruno Fernandes", "manager": false, "timeframe": "last_game", "opponent": null}
- "Who is our current manager?"
  --> {"sub_intent": "manager_stats", "player_name": null, "manager": true, "timeframe": "overall", "opponent": null}
- "What were the match stats against Fulham?"
  --> {"sub_intent": "match_stats", "player_name": null, "manager": false, "timeframe": "last_game", "opponent": "Fulham"}
- "Where are we in the Premier League table?"
  --> {"sub_intent": "standings", "player_name": null, "manager": false, "timeframe": "season", "opponent": null}

Respond ONLY with valid JSON. No markdown wrappers.
"""


def extract_stats_subintent(message: str) -> dict:
    """Parse user query into structured stats sub-intent."""
    print("\n--- [DEBUG: STATS SUB-INTENT CLASSIFICATION] ---")
    print(f"Input Message: '{message}'")
    try:
        resp = _client().messages.create(
            model=config.MODEL_HAIKU,
            temperature=0,
            max_tokens=150,
            system=_STATS_SUBINTENT_PROMPT,
            messages=[
                {"role": "user", "content": message},
            ],
        )
        raw = resp.content[0].text.strip()
        print(f"Raw LLM Response: {raw}")
        # Clean any markdown block formatting
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n|\n```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        print(f"Sub-Intent: '{result.get('sub_intent')}', Player: '{result.get('player_name')}', Timeframe: '{result.get('timeframe')}'")
        print("--------------------------------------------------\n")
        return result
    except Exception as exc:
        logger.warning("[StatsHandler] Subintent extraction failed (%s)", exc)
        print(f"Classification Error: {exc} — defaulting to general")
        print("--------------------------------------------------\n")
        return {
            "sub_intent": "general",
            "player_name": None,
            "manager": False,
            "timeframe": None,
            "opponent": None,
        }


# ---------------------------------------------------------------------------
# Step 3 — Data Fetchers (Standings, Match Stats, Player Stats)
# ---------------------------------------------------------------------------
def get_standings_context() -> str:
    """Fetch and format current Premier League standings."""
    api_key = (config.FOOTBALL_API_KEY or "").strip()
    headers = {"X-Auth-Token": api_key} if api_key else {}
    url = f"{config.FOOTBALL_API_BASE}/competitions/PL/standings"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        standings_tables = data.get("standings", [])
        if not standings_tables:
            return "[Standings Context]\nNo standings data available at the moment."

        table = standings_tables[0].get("table", [])
        lines = ["[Premier League Standings Context]"]
        season = data.get("season", {}).get("startDate", "")[:4]
        if season:
            lines.append(f"Season: {season}/{int(season)+1}")

        lines.append(f"{'Pos':<4}{'Team':<22}{'Pld':<6}{'W':<4}{'D':<4}{'L':<4}{'GD':<6}{'Pts':<5}")
        lines.append("-" * 55)

        for row in table:
            pos = row.get("position")
            team_name = row.get("team", {}).get("name", "")
            played = row.get("playedGames")
            won = row.get("won")
            draw = row.get("draw")
            lost = row.get("lost")
            gd = row.get("goalDifference")
            pts = row.get("points")

            marker = " <== Manchester United" if row.get("team", {}).get("id") == MAN_UTD_ID else ""
            lines.append(
                f"{pos:<4}{team_name:<22}{played:<6}{won:<4}{draw:<4}{lost:<4}{gd:<+6}{pts:<5}{marker}"
            )

        return "\n".join(lines)
    except Exception as exc:
        logger.error("[StatsHandler] Standings fetch failed: %s", exc)
        return "[Standings Context]\nUnable to fetch current Premier League standings right now."


def get_match_stats_context(opponent: Optional[str] = None) -> str:
    """Fetch completed match statistics from live APIs (TheSportsDB / Football-Data API)."""
    # 1. Try TheSportsDB live API first
    try:
        from .match_recap import fetch_recent_finished_matches
        finished_matches = fetch_recent_finished_matches()
        if finished_matches:
            target_match = None
            if opponent:
                opp_clean = opponent.lower().strip()
                for m in finished_matches:
                    home = m.get("strHomeTeam", "").lower()
                    away = m.get("strAwayTeam", "").lower()
                    if opp_clean in home or opp_clean in away:
                        target_match = m
                        break

            if not target_match:
                target_match = finished_matches[0]

            home = target_match.get("strHomeTeam")
            away = target_match.get("strAwayTeam")
            score_h = target_match.get("intHomeScore")
            score_a = target_match.get("intAwayScore")
            date = target_match.get("dateEvent")
            event_name = target_match.get("strEvent")

            return (
                f"[Match Stats Context]\n"
                f"Match: {event_name}\n"
                f"Date: {date}\n"
                f"Full-Time Score: {home} {score_h} - {score_a} {away}\n"
                f"Status: Finished"
            )
    except Exception as exc:
        logger.warning("[StatsHandler] Fetching finished matches from TheSportsDB failed: %s", exc)

    # 2. Fallback to Football-Data API finished matches
    api_key = (config.FOOTBALL_API_KEY or "").strip()
    headers = {"X-Auth-Token": api_key} if api_key else {}
    url = f"{config.FOOTBALL_API_BASE}/teams/{MAN_UTD_ID}/matches?status=FINISHED"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            api_matches = resp.json().get("matches", [])
            if api_matches:
                last_m = api_matches[-1]
                home = last_m.get("homeTeam", {}).get("name")
                away = last_m.get("awayTeam", {}).get("name")
                score_h = last_m.get("score", {}).get("fullTime", {}).get("home")
                score_a = last_m.get("score", {}).get("fullTime", {}).get("away")
                ht_h = last_m.get("score", {}).get("halfTime", {}).get("home")
                ht_a = last_m.get("score", {}).get("halfTime", {}).get("away")
                date = last_m.get("utcDate", "")[:10]

                return (
                    f"[Match Stats Context]\n"
                    f"Match: {home} vs {away}\n"
                    f"Date: {date}\n"
                    f"Full-Time Score: {home} {score_h} - {score_a} {away}\n"
                    f"Half-Time Score: {ht_h} - {ht_a}\n"
                    f"Status: Finished"
                )
    except Exception as e:
        logger.warning("[StatsHandler] API match fetch failed: %s", e)

    return "[Match Stats Context]\nNo recent match statistics found."


def get_player_or_manager_context(parsed: dict) -> str:
    """Validate entity squad membership and retrieve goals/stats context."""
    player_name = parsed.get("player_name")
    is_manager_query = parsed.get("manager", False)
    timeframe = parsed.get("timeframe", "overall")

    if is_manager_query:
        _, manager, _ = validate_squad_member("manager")
        if not manager:
            sync_squad_data()
            _, manager, _ = validate_squad_member("manager")

        if manager:
            contract = manager.get("contract") or {}
            c_start = contract.get("start") or "N/A"
            c_until = contract.get("until") or "N/A"
            return (
                f"[Manager Context]\n"
                f"Name: {manager.get('name')}\n"
                f"Nationality: {manager.get('nationality', 'N/A')}\n"
                f"Date of Birth: {manager.get('date_of_birth', 'N/A')}\n"
                f"Contract: {c_start} to {c_until}"
            )
        return "[Manager Context]\nUnable to fetch manager details."

    if not player_name:
        return "[Player Stats Context]\nNo player name specified in the query."

    # Validate player squad membership
    is_in_squad, player_record, entity_type = validate_squad_member(player_name)

    if not is_in_squad:
        logger.info("[StatsHandler] Player '%s' not in squad -> return out-of-scope response", player_name)
        return (
            f"[Squad Scope Notice]\n"
            f"Note: '{player_name}' is NOT currently in Manchester United's official squad. "
            f"Inform the user warmly that as a Manchester United fan companion, you track stats for Red Devils players."
        )

    official_name = player_record.get("name", player_name)
    pos = player_record.get("position", "Player")
    nat = player_record.get("nationality", "")

    lines = [
        f"[Player Stats Context]",
        f"Player: {official_name}",
        f"Position: {pos}",
        f"Nationality: {nat}",
        f"Club: Manchester United",
    ]

    # Fetch last game goals if applicable
    records_file = config.DATA_DIR / "records.json"
    if records_file.exists():
        try:
            data = json.loads(records_file.read_text(encoding="utf-8"))
            completed = [r for r in data if r.get("type") == "fixture" and r.get("status") == "completed"]
            if completed:
                last_m = completed[-1]
                scorers = last_m.get("result", {}).get("scorers", [])
                p_short = official_name.split()[-1]
                matching_goals = [s for s in scorers if p_short.lower() in s.lower()]
                lines.append(
                    f"Last Game ({last_m.get('opponent')} on {last_m.get('date')}): "
                    f"{len(matching_goals)} goal(s) ({', '.join(matching_goals) if matching_goals else 'none'})"
                )
        except Exception as e:
            logger.warning("[StatsHandler] Failed reading match goals: %s", e)

    # Fetch season goals from Football-Data top scorers for recent seasons
    try:
        api_key = (config.FOOTBALL_API_KEY or "").strip()
        headers = {"X-Auth-Token": api_key} if api_key else {}
        p_short = official_name.split()[-1].lower()
        season_found = False

        for season_yr in ["2025", "2024"]:
            url = f"{config.FOOTBALL_API_BASE}/competitions/PL/scorers?season={season_yr}&limit=100"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                scorers_list = resp.json().get("scorers", [])
                p_match = next(
                    (s for s in scorers_list if p_short in s.get("player", {}).get("name", "").lower()),
                    None
                )
                if p_match:
                    s_label = f"{season_yr}/{int(season_yr)+1}"
                    goals = p_match.get("goals", 0)
                    assists = p_match.get("assists", 0)
                    matches = p_match.get("playedMatches", 0)
                    lines.append(
                        f"Premier League Season {s_label} Stats: {goals} goal(s), {assists} assist(s) in {matches} match(es)"
                    )
                    season_found = True

        if not season_found:
            lines.append("Premier League Season Stats: 0 goals recorded in top scorers table")
    except Exception as e:
        logger.warning("[StatsHandler] Top scorers fetch failed: %s", e)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 4 — Check Cached Match Recap for Last Game Queries
# ---------------------------------------------------------------------------
def check_cached_recap_for_stats(message: str, parsed: dict) -> Optional[str]:
    """
    Check if the query asks about last game events/scorers/stats AND if a cached match recap exists.
    If cached recap exists and answers the query, return the answer text.
    Otherwise return None to fallback to live_stats_lookup API.
    """
    timeframe = parsed.get("timeframe")
    sub_intent = parsed.get("sub_intent")
    query_lower = message.lower()

    # Trigger check if query relates to last game / match events / goals
    is_last_game_query = (
        timeframe == "last_game"
        or sub_intent in ("match_stats", "player_stats")
        or any(w in query_lower for w in ("last game", "last match", "who scored", "goals in the last", "previous game", "previous match", "scored in"))
    )

    if not is_last_game_query:
        return None

    print(f"\n--- [DEBUG STATS: RECAP CACHE CHECK] ---", flush=True)
    print(f"Checking if cached match recap can answer query: '{message}'", flush=True)
    try:
        from .match_recap import fetch_recent_finished_matches, _check_recap_cache
        matches = fetch_recent_finished_matches()
        if not matches:
            print("[DEBUG STATS] No recent finished matches returned. Falling back to live_stats_lookup API.", flush=True)
            return None

        last_match = matches[0]
        match_id = last_match.get("idEvent", "")
        if not match_id:
            return None

        cached_recap = _check_recap_cache(match_id)
        if not cached_recap:
            print(f"[DEBUG STATS] Cache MISS for last match ID '{match_id}' ({last_match.get('strEvent')}). Falling back to live_stats_lookup API.\n", flush=True)
            return None

        print(f"[DEBUG STATS] CACHE HIT! Found cached recap for '{last_match.get('strEvent')}'. Evaluating if recap answers query...", flush=True)

        # Ask LLM if the cached recap contains the answer
        prompt = (
            f"You are an assistant answering a Manchester United match query using a CACHED MATCH RECAP.\n\n"
            f"MATCH RECAP:\n{cached_recap}\n\n"
            f"USER QUERY:\n{message}\n\n"
            "Can the user's question be accurately answered from the cached match recap above?\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "can_answer": true or false,\n'
            '  "answer": "A concise, direct 1-2 sentence answer to the user query focusing ONLY on what was asked (e.g. listing goal scorers or scoreline). Do not include full match recaps or unrelated match sections." (or "" if false)\n'
            "}"
        )
        resp = _client().messages.create(
            model=config.MODEL_HAIKU,
            temperature=0,
            max_tokens=200,
            system="You check if a match recap answers a query.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n|\n```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)

        if result.get("can_answer") and result.get("answer"):
            answer = result["answer"]
            print(f"[DEBUG STATS] SUCCESS! Query answered directly from cached recap:\n{answer}\n---------------------------------------\n", flush=True)
            return f"[Context: Answered from Cached Match Recap for {last_match.get('strEvent')}]\n{answer}"
        else:
            print(f"[DEBUG STATS] Cached recap did not contain sufficient details to answer query. Falling back to live_stats_lookup API.\n---------------------------------------\n", flush=True)
            return None
    except Exception as exc:
        logger.warning("[StatsHandler] Failed checking cached recap: %s", exc)
        print(f"[DEBUG STATS] Error checking recap cache: {exc}. Falling back to live_stats_lookup API.\n---------------------------------------\n", flush=True)
        return None


# ---------------------------------------------------------------------------
# Step 5 — Main Entry Point
# ---------------------------------------------------------------------------
def handle_stats_intent(message: str) -> str:
    """
    Main handler for the 'stats' intent.
    Extracts sub-intent, checks cached recap for last game queries first,
    validates squad membership, fetches context, and returns formatted context block.
    """
    logger.info("[StatsHandler] Handling stats message: '%s'", message)

    parsed = extract_stats_subintent(message)

    # 1. Check cached match recap first for last game / match event queries
    cached_recap_answer = check_cached_recap_for_stats(message, parsed)
    if cached_recap_answer:
        return cached_recap_answer

    # 2. Fall back to live_stats_lookup / API / records.json
    sub_intent = parsed.get("sub_intent")
    if sub_intent == "standings":
        return get_standings_context()
    elif sub_intent == "match_stats":
        return get_match_stats_context(parsed.get("opponent"))
    elif sub_intent in ("player_stats", "manager_stats"):
        return get_player_or_manager_context(parsed)
    else:
        # General stats query fallback — combine standings summary and last match stats
        match_ctx = get_match_stats_context()
        return f"{match_ctx}"
