"""Match Reminders & Calendar Handler for Manchester United Fan Companion.

Flow:
  handle_calendar_intent(message, user_id)
        ↓
  extract_calendar_intent() — LLM mini-call → sub-intent (set_reminder, confirm_action, cancel_action, fetch_reminders, remove_reminder)
        ↓
  Execute workflow:
    - set_reminder      → fixture lookup → set pending confirmation state → prompt user for confirmation
    - confirm_action    → execute pending creation (via Google Calendar API & save reminders.json) or deletion
    - cancel_action     → clear pending state
    - fetch_reminders   → read data/reminders.json → return formatted list
    - remove_reminder   → locate reminder → set pending deletion state → prompt user for confirmation before deleting
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from . import config
from .fixtures_handler import (
    extract_fixture_intent,
    fetch_team_matches,
    filter_matches,
    _format_datetime,
    MAN_UTD_ID,
)
from .calendar_service import create_calendar_event, delete_calendar_event

logger = logging.getLogger(__name__)

REMINDERS_FILE = config.DATA_DIR / "reminders.json"

# In-memory session tracking for pending confirmation states per user
# Schema: { user_id: { "action": "create_reminder"|"delete_reminder", "fixture": dict, "reminder": dict } }
_PENDING_CONFIRMATIONS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Data Persistence (reminders.json)
# ---------------------------------------------------------------------------
def load_reminders() -> List[Dict[str, Any]]:
    """Load all saved match reminders from data/reminders.json."""
    if not REMINDERS_FILE.exists():
        return []
    try:
        data = REMINDERS_FILE.read_text(encoding="utf-8").strip()
        if not data:
            return []
        return json.loads(data)
    except Exception as exc:
        logger.error("[RemindersHandler] Failed to load reminders.json: %s", exc)
        return []


def save_reminders(reminders: List[Dict[str, Any]]) -> None:
    """Save the match reminders list back to data/reminders.json."""
    try:
        REMINDERS_FILE.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
        print(f"[DEBUG: REMINDERS HANDLER] Saved {len(reminders)} reminders to {REMINDERS_FILE.name}")
    except Exception as exc:
        logger.error("[RemindersHandler] Failed to save reminders.json: %s", exc)


# ---------------------------------------------------------------------------
# Pending State Helpers
# ---------------------------------------------------------------------------
def get_pending_confirmation(user_id: str) -> Optional[Dict[str, Any]]:
    return _PENDING_CONFIRMATIONS.get(user_id)


def set_pending_confirmation(user_id: str, pending_data: Dict[str, Any]) -> None:
    _PENDING_CONFIRMATIONS[user_id] = pending_data


def clear_pending_confirmation(user_id: str) -> None:
    _PENDING_CONFIRMATIONS.pop(user_id, None)


# ---------------------------------------------------------------------------
# LLM Sub-Intent Extraction
# ---------------------------------------------------------------------------
_CALENDAR_INTENT_PROMPT = """\
You are an intent classifier for a Manchester United match reminder & calendar system.
The user is asking a calendar-related question or responding to a previous confirmation prompt.

Return a JSON object with exactly two fields:
1. "sub_intent" — one of:
   - "confirm_action"             : user explicitly confirming a pending action (e.g. "yes", "confirm", "do it", "sure", "ok", "please set it", "delete it", "yes delete it")
   - "cancel_action"              : user cancelling a pending action (e.g. "no", "cancel", "don't", "stop", "never mind", "keep it")
   - "set_reminder"               : user asking to set, add, or create a calendar reminder for EXACTLY ONE match
   - "multiple_reminders_rejected": user asking to set reminders for MULTIPLE matches at once (e.g. "set reminders for next 3 games", "remind me about Arsenal and Chelsea", "add all home games", "set reminders for all upcoming matches")
   - "fetch_reminders"            : user asking to show, list, view, or check existing set match reminders
   - "remove_reminder"            : user asking to delete, cancel, or remove an existing match reminder

2. "target_opponent" — opponent team name mentioned in query (e.g. "Arsenal", "Chelsea"), or "" if not applicable.

State & Multiple Query Rules:
- If the user asks to set reminders for more than one match at a time (e.g. multiple team names, "next N games", "all home games", "all matches"), ALWAYS classify as "multiple_reminders_rejected".
- If HAS_PENDING_ACTION is true AND user message is an affirmative response ("yes", "yep", "confirm", "sure", "ok", "do it", "go ahead"), classify as "confirm_action".
- If HAS_PENDING_ACTION is true AND user message is a negative response ("no", "cancel", "don't", "stop"), classify as "cancel_action".

Respond ONLY with valid JSON. No markdown, no explanation.

Examples:
  User: "Remind me about the Arsenal game"               → {"sub_intent": "set_reminder", "target_opponent": "Arsenal"}
  User: "Set reminders for the next 3 matches"          → {"sub_intent": "multiple_reminders_rejected", "target_opponent": ""}
  User: "Remind me about the Arsenal and Chelsea games"  → {"sub_intent": "multiple_reminders_rejected", "target_opponent": ""}
  User: "Add all home matches to my calendar"           → {"sub_intent": "multiple_reminders_rejected", "target_opponent": ""}
  User: "Show my saved match reminders"                  → {"sub_intent": "fetch_reminders", "target_opponent": ""}
  User: "Delete my reminder for the Chelsea game"       → {"sub_intent": "remove_reminder", "target_opponent": "Chelsea"}
  User: "Yes" (when HAS_PENDING_ACTION=true)              → {"sub_intent": "confirm_action", "target_opponent": ""}
  User: "Cancel" (when HAS_PENDING_ACTION=true)           → {"sub_intent": "cancel_action", "target_opponent": ""}
"""


def extract_calendar_intent(message: str, user_id: str = "default_user", has_pending: Optional[bool] = None) -> dict:
    """Classify user calendar query into structured sub-intent using Anthropic MODEL_HAIKU."""
    from .main import _client
    if has_pending is None:
        has_pending = get_pending_confirmation(user_id) is not None
    system_prompt = f"HAS_PENDING_ACTION: {str(has_pending).lower()}\n\n" + _CALENDAR_INTENT_PROMPT
    try:
        resp = _client().messages.create(
            model=config.MODEL_HAIKU,
            temperature=0,
            max_tokens=80,
            system=system_prompt,
            messages=[
                {"role": "user", "content": message},
            ],
        )
        raw = resp.content[0].text.strip()
        if "```" in raw:
            lines = [line.strip() for line in raw.splitlines() if not line.strip().startswith("```")]
            raw = "".join(lines).strip()
        result = json.loads(raw)
        sub_intent = result.get("sub_intent", "fetch_reminders")
        target_opponent = result.get("target_opponent", "")
        print(f"[DEBUG: CALENDAR INTENT] sub_intent='{sub_intent}', opponent='{target_opponent}', has_pending={has_pending}")
        return {"sub_intent": sub_intent, "target_opponent": target_opponent}
    except Exception as exc:
        logger.warning("[RemindersHandler] Sub-intent classification error: %s", exc)
        # Simple fallback heuristics if LLM mini-call fails
        msg_lower = message.lower().strip()
        if has_pending is None:
            has_pending = get_pending_confirmation(user_id) is not None
        if not has_pending and any(w in msg_lower for w in ("next 2", "next 3", "next 4", "next 5", "all games", "all matches", "all home", "all away", " and ")):
            if "remind" in msg_lower or "calendar" in msg_lower or "set" in msg_lower or "add" in msg_lower:
                return {"sub_intent": "multiple_reminders_rejected", "target_opponent": ""}
        if has_pending and any(w in msg_lower for w in ("yes", "yep", "yeah", "confirm", "sure", "ok", "do it", "set it", "go ahead", "please")):
            return {"sub_intent": "confirm_action", "target_opponent": ""}
        if has_pending and any(w in msg_lower for w in ("no", "nope", "cancel", "stop", "don't", "dont", "abort")):
            return {"sub_intent": "cancel_action", "target_opponent": ""}
        if any(w in msg_lower for w in ("show", "view", "list", "fetch", "what reminders")):
            return {"sub_intent": "fetch_reminders", "target_opponent": ""}
        if any(w in msg_lower for w in ("delete", "remove", "clear")):
            return {"sub_intent": "remove_reminder", "target_opponent": ""}
        return {"sub_intent": "set_reminder", "target_opponent": ""}


# ---------------------------------------------------------------------------
# Public Handler Entrypoint
# ---------------------------------------------------------------------------
def handle_calendar_intent(message: str, user_id: str = "default_user") -> str:
    """
    Main orchestrator for match reminders and calendar events.
    """
    print(f"\n==================================================", flush=True)
    print(f"[DEBUG: CALENDAR HANDLER] Incoming Message: '{message}'", flush=True)
    print(f"[DEBUG: CALENDAR HANDLER] User ID: '{user_id}'", flush=True)
    
    pending = get_pending_confirmation(user_id)
    has_pending = pending is not None
    if has_pending:
        print(f"[DEBUG: PENDING STATE] Active Pending Action Found for user '{user_id}': {pending.get('action')}", flush=True)
    else:
        print(f"[DEBUG: PENDING STATE] No active pending action for user '{user_id}'", flush=True)

    intent_res = extract_calendar_intent(message, user_id=user_id, has_pending=has_pending)
    sub_intent = intent_res["sub_intent"]
    target_opponent = intent_res["target_opponent"]
    print(f"[DEBUG: CALENDAR SUB-INTENT] Classified: sub_intent='{sub_intent}', target_opponent='{target_opponent}'", flush=True)
    print(f"==================================================\n", flush=True)

    # -----------------------------------------------------------------------
    # Step A: Handle Pending Action Confirmation / Cancellation
    # -----------------------------------------------------------------------
    if has_pending:
        if sub_intent == "confirm_action":
            action_type = pending.get("action")
            print(f"[DEBUG: ACTION CONFIRMED] User '{user_id}' confirmed pending action: '{action_type}'", flush=True)
            
            if action_type == "create_reminder":
                fixture = pending.get("fixture", {})
                print(f"[DEBUG: CREATING REMINDER] Creating Google Calendar event for fixture: {fixture.get('home_team')} vs {fixture.get('away_team')}", flush=True)
                
                # Create Google Calendar event
                cal_result = create_calendar_event(fixture)
                event_id = cal_result.get("event_id", "")
                calendar_url = cal_result.get("html_link") or cal_result.get("web_url") or ""

                home = fixture.get("home_team", "Manchester United")
                away = fixture.get("away_team", "Opponent")
                fixture_name = f"{home} vs {away}"
                
                reminder_record = {
                    "id": f"rem_{uuid.uuid4().hex[:8]}",
                    "fixture_name": fixture_name,
                    "opponent": fixture.get("opponent", away),
                    "competition": fixture.get("competition", "Match"),
                    "date_utc": fixture.get("utc_date", ""),
                    "date_ist": fixture.get("date_ist", ""),
                    "venue": fixture.get("venue", "Old Trafford"),
                    "event_id": event_id,
                    "calendar_url": calendar_url,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                reminders = load_reminders()
                # Deduplicate by fixture_name and date_utc
                reminders = [r for r in reminders if r.get("fixture_name") != fixture_name or r.get("date_utc") != fixture.get("utc_date")]
                reminders.append(reminder_record)
                save_reminders(reminders)

                clear_pending_confirmation(user_id)
                print(f"[DEBUG: REMINDER STORED] Saved '{fixture_name}' to reminders.json. Total count: {len(reminders)}", flush=True)

                link_md = f"[📅 View / Add to Google Calendar]({calendar_url})" if calendar_url else ""
                return (
                    f"✅ **Calendar Event & Reminder Set!**\n\n"
                    f"• **Fixture:** {fixture_name}\n"
                    f"• **Competition:** {fixture.get('competition')}\n"
                    f"• **Kickoff:** {fixture.get('date_ist')}\n"
                    f"• **Venue:** {fixture.get('venue')}\n\n"
                    f"{link_md}"
                )

            elif action_type == "delete_reminder":
                rem_to_delete = pending.get("reminder", {})
                event_id = rem_to_delete.get("event_id", "")
                fixture_name = rem_to_delete.get("fixture_name", "Match")
                print(f"[DEBUG: DELETING REMINDER] Confirmed deletion for '{fixture_name}' (Event ID: '{event_id}')", flush=True)

                # Call Google Calendar API to delete event
                if event_id:
                    delete_calendar_event(event_id)

                # Remove from reminders.json
                reminders = load_reminders()
                reminders = [r for r in reminders if r.get("id") != rem_to_delete.get("id")]
                save_reminders(reminders)

                clear_pending_confirmation(user_id)
                print(f"[DEBUG: REMINDER REMOVED] Deleted '{fixture_name}' from reminders.json. Remaining count: {len(reminders)}", flush=True)
                return f"🗑️ The reminder for **{fixture_name}** has been deleted from your Google Calendar and saved reminders."

        elif sub_intent == "cancel_action":
            action_type = pending.get("action")
            print(f"[DEBUG: ACTION CANCELLED] User '{user_id}' cancelled pending action: '{action_type}'. State cleared.", flush=True)
            clear_pending_confirmation(user_id)
            return "❌ Action cancelled. No changes were made to your calendar or reminders."

    # -----------------------------------------------------------------------
    # Step B: New Sub-intents (Set Reminder, Multiple Rejected, Fetch Reminders, Remove Reminder)
    # -----------------------------------------------------------------------

    if sub_intent == "multiple_reminders_rejected":
        print(f"[DEBUG: MULTIPLE REMINDERS REJECTED] Query requested multiple reminders. Refusing batch creation.", flush=True)
        return "I can only set a reminder for **one match at a time**! Please specify a single match you'd like a reminder for (e.g. *'Set a reminder for the next Arsenal match'*)."

    # 1. SET REMINDER — Fixture Lookup & Confirmation Request
    elif sub_intent == "set_reminder":
        print(f"[DEBUG: SET REMINDER] Searching upcoming fixtures for query...", flush=True)
        # Extract fixture sub-intent using existing fixture extractor
        fix_intent = extract_fixture_intent(message)
        sub_fix_intent = fix_intent["sub_intent"]
        opponent_name = fix_intent["opponent_name"] or target_opponent
        print(f"[DEBUG: FIXTURE LOOKUP] Extracted sub_intent='{sub_fix_intent}', opponent='{opponent_name}'", flush=True)

        matches = fetch_team_matches(future=True)
        filtered = filter_matches(matches, sub_fix_intent, opponent_name)

        if not filtered:
            # Fallback: search all future matches for opponent substring
            if opponent_name:
                filtered = [
                    m for m in matches
                    if opponent_name.lower() in (m.get("homeTeam") or {}).get("name", "").lower()
                    or opponent_name.lower() in (m.get("awayTeam") or {}).get("name", "").lower()
                ]

        if not filtered:
            opp_str = f" against **{opponent_name}**" if opponent_name else ""
            print(f"[DEBUG: FIXTURE LOOKUP] No upcoming match found for opponent '{opponent_name}'", flush=True)
            return f"I couldn't find an upcoming Manchester United match{opp_str}. Please verify the team name or fixture schedule."

        target_match = filtered[0]
        home_name = (target_match.get("homeTeam") or {}).get("name", "Unknown")
        away_name = (target_match.get("awayTeam") or {}).get("name", "Unknown")
        competition = (target_match.get("competition") or {}).get("name", "Premier League")
        venue = target_match.get("venue") or ("Old Trafford" if target_match.get("homeTeam", {}).get("id") == MAN_UTD_ID else "Away Stadium")
        utc_date = target_match.get("utcDate", "")
        date_ist = _format_datetime(utc_date)

        print(f"[DEBUG: FIXTURE FOUND] {home_name} vs {away_name} ({competition}) on {date_ist} at {venue}", flush=True)

        fixture_dict = {
            "home_team": home_name,
            "away_team": away_name,
            "opponent": away_name if "manchester" in home_name.lower() or "man utd" in home_name.lower() else home_name,
            "competition": competition,
            "venue": venue,
            "utc_date": utc_date,
            "date_ist": date_ist,
        }

        # Save pending creation state
        set_pending_confirmation(user_id, {
            "action": "create_reminder",
            "fixture": fixture_dict,
        })
        print(f"[DEBUG: PENDING CONFIRMATION SET] Waiting for user '{user_id}' to confirm reminder creation.", flush=True)

        return (
            f"I found the upcoming fixture:\n\n"
            f"• **Match:** {home_name} vs {away_name}\n"
            f"• **Competition:** {competition}\n"
            f"• **Kickoff:** {date_ist}\n"
            f"• **Venue:** {venue}\n\n"
            f"Would you like me to confirm and set a Google Calendar reminder for this match? "
            f"(Reply **Yes** or **Confirm** to set it, or **Cancel** to abort)"
        )

    # 2. FETCH REMINDERS
    elif sub_intent == "fetch_reminders":
        reminders = load_reminders()
        print(f"[DEBUG: FETCH REMINDERS] Retrieved {len(reminders)} saved reminders from reminders.json", flush=True)
        if not reminders:
            return "You currently have no saved match reminders."

        lines = ["📌 **Your Saved Match Reminders:**\n"]
        for idx, r in enumerate(reminders, 1):
            fixture_name = r.get("fixture_name", "Match")
            comp = r.get("competition", "")
            date_ist = r.get("date_ist", "")
            venue = r.get("venue", "")
            url = r.get("calendar_url", "")
            link_str = f" — [📅 Calendar Link]({url})" if url else ""

            lines.append(f"{idx}. **{fixture_name}** ({comp})")
            lines.append(f"   • Kickoff: {date_ist}")
            lines.append(f"   • Venue: {venue}{link_str}\n")

        lines.append("To remove a reminder, simply ask e.g. *'Delete my Arsenal reminder'*.")
        return "\n".join(lines)

    # 3. REMOVE REMINDER — 2-Step Confirmation Request
    elif sub_intent == "remove_reminder":
        reminders = load_reminders()
        print(f"[DEBUG: REMOVE REMINDER] Searching {len(reminders)} saved reminders for opponent '{target_opponent}'...", flush=True)
        if not reminders:
            return "You currently have no saved match reminders to remove."

        target_rem = None
        if target_opponent:
            for r in reminders:
                if target_opponent.lower() in r.get("opponent", "").lower() or target_opponent.lower() in r.get("fixture_name", "").lower():
                    target_rem = r
                    break

        if not target_rem and len(reminders) == 1:
            target_rem = reminders[0]

        if not target_rem:
            print(f"[DEBUG: REMOVE REMINDER] Target reminder ambiguous or multiple exist. Prompting user to specify.", flush=True)
            lines = ["I found multiple saved reminders. Which one would you like to delete?\n"]
            for r in reminders:
                lines.append(f"• **{r.get('fixture_name')}** ({r.get('date_ist')})")
            lines.append("\nPlease specify the opponent name (e.g. *'Remove the Arsenal reminder'*).")
            return "\n".join(lines)

        fixture_name = target_rem.get("fixture_name", "Match")
        date_ist = target_rem.get("date_ist", "")
        print(f"[DEBUG: REMOVE REMINDER FOUND] Target: '{fixture_name}'. Setting 2-step pending deletion state.", flush=True)

        # Save pending deletion state
        set_pending_confirmation(user_id, {
            "action": "delete_reminder",
            "reminder": target_rem,
        })

        return (
            f"Are you sure you want to delete the reminder for **{fixture_name}** ({date_ist}) "
            f"from your Google Calendar and saved reminders?\n\n"
            f"(Reply **Yes** or **Confirm** to delete, or **Cancel** to keep it)"
        )

    return "I can help you set match reminders, list your saved reminders, or delete reminders. What would you like to do?"
