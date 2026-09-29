"""Google Calendar Service module for Manchester United Fan Companion.

Handles authentication, event creation, event deletion, and web link generation
via Google Calendar API v3.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
import urllib.parse

from . import config

logger = logging.getLogger(__name__)

# Scopes required for Google Calendar API
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def get_calendar_service():
    """
    Build and return a Google Calendar API service client.
    
    Checks token.json for saved credentials; if missing or expired,
    attempts to load credentials.json and refresh/authorize.
    Returns None if credentials are missing or authorization fails.
    """
    creds_path = getattr(config, "GOOGLE_CALENDAR_CREDS_PATH", "credentials.json")
    token_path = getattr(config, "GOOGLE_CALENDAR_TOKEN_PATH", "token.json")

    creds = None
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as ref_exc:
                    logger.warning("[GoogleCalendar] Token refresh failed: %s", ref_exc)
                    creds = None

            if not creds and os.path.exists(creds_path):
                if os.getenv("SKIP_INTERACTIVE_OAUTH") == "1":
                    logger.info("[GoogleCalendar] SKIP_INTERACTIVE_OAUTH set — using web link generator fallback.")
                    creds = None
                else:
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                        try:
                            creds = flow.run_local_server(port=8080, open_browser=True)
                        except Exception:
                            creds = flow.run_local_server(port=0, open_browser=True)
                    except Exception as flow_exc:
                        logger.warning("[GoogleCalendar] Interactive authorization skipped: %s", flow_exc)
                        creds = None

            if creds:
                try:
                    with open(token_path, "w", encoding="utf-8") as f:
                        f.write(creds.to_json())
                except Exception as save_exc:
                    logger.warning("[GoogleCalendar] Could not save token.json: %s", save_exc)

        if creds:
            return build("calendar", "v3", credentials=creds)
    except Exception as exc:
        logger.warning("[GoogleCalendar] Failed to initialize Google Calendar service: %s", exc)
        print(f"[DEBUG: GOOGLE CALENDAR] API Client initialization warning: {exc}")

    return None


def generate_google_calendar_url(fixture: dict) -> str:
    """
    Generate a direct Google Calendar Web 'Add to Calendar' URL.
    
    Example template URL:
    https://calendar.google.com/calendar/render?action=TEMPLATE&text=...&dates=...&details=...&location=...
    """
    home = fixture.get("home_team", "Manchester United")
    away = fixture.get("away_team", "Opponent")
    comp = fixture.get("competition", "Match")
    venue = fixture.get("venue", "Old Trafford")
    date_utc = fixture.get("utc_date", "")

    title = f"{home} vs {away}"
    details = f"Manchester United Fan Companion Reminder\nCompetition: {comp}\nFixture: {home} vs {away}\nVenue: {venue}"

    # Parse start and end time (default to 2 hour duration)
    try:
        dt_start = datetime.fromisoformat(date_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        dt_start = datetime.now(timezone.utc) + timedelta(days=1)
    
    dt_end = dt_start + timedelta(hours=2)

    # Format datetimes in YYYYMMDDTHHMMSSZ
    start_str = dt_start.strftime("%Y%m%dT%H%M%SZ")
    end_str = dt_end.strftime("%Y%m%dT%H%M%SZ")
    dates_param = f"{start_str}/{end_str}"

    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": dates_param,
        "details": details,
        "location": venue,
    }
    return f"https://calendar.google.com/calendar/render?{urllib.parse.urlencode(params)}"


def create_calendar_event(fixture: dict) -> dict:
    """
    Create an event in Google Calendar using Google Calendar API v3.
    
    Falls back gracefully to web URL link if API client is unauthorized.
    """
    calendar_id = getattr(config, "GOOGLE_CALENDAR_ID", "primary")
    web_url = generate_google_calendar_url(fixture)

    home = fixture.get("home_team", "Manchester United")
    away = fixture.get("away_team", "Opponent")
    comp = fixture.get("competition", "Match")
    venue = fixture.get("venue", "Old Trafford")
    date_utc = fixture.get("utc_date", "")

    summary = f"{home} vs {away}"
    description = f"Manchester United Fan Companion Reminder\nCompetition: {comp}\nFixture: {home} vs {away}\nVenue: {venue}"

    try:
        dt_start = datetime.fromisoformat(date_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        dt_start = datetime.now(timezone.utc) + timedelta(days=1)
    
    dt_end = dt_start + timedelta(hours=2)

    event_body = {
        "summary": summary,
        "location": venue,
        "description": description,
        "start": {
            "dateTime": dt_start.isoformat(),
            "timeZone": "UTC",
        },
        "end": {
            "dateTime": dt_end.isoformat(),
            "timeZone": "UTC",
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 60},
                {"method": "popup", "minutes": 15},
            ],
        },
    }

    print(f"\n--- [DEBUG: GOOGLE CALENDAR API] ---", flush=True)
    print(f"Creating Event: '{summary}' ({comp})", flush=True)
    print(f"Kickoff (UTC): {dt_start.isoformat()} | Venue: {venue}", flush=True)

    service = get_calendar_service()
    if service:
        try:
            created_event = service.events().insert(calendarId=calendar_id, body=event_body).execute()
            event_id = created_event.get("id", "")
            html_link = created_event.get("htmlLink", web_url)
            print(f"[DEBUG: GOOGLE CALENDAR API] SUCCESS! Created Event ID: '{event_id}'", flush=True)
            print(f"[DEBUG: GOOGLE CALENDAR API] Link: {html_link}\n-----------------------------------\n", flush=True)
            return {
                "status": "ok",
                "api_used": True,
                "event_id": event_id,
                "html_link": html_link,
                "web_url": web_url,
            }
        except Exception as exc:
            logger.error("[GoogleCalendar] API insert event failed: %s", exc)
            print(f"[DEBUG: GOOGLE CALENDAR API] API Insert Failed: {exc}. Falling back to web link.\n-----------------------------------\n", flush=True)

    print(f"[DEBUG: GOOGLE CALENDAR API] API Client unavailable. Generated Direct Web URL: {web_url}\n-----------------------------------\n", flush=True)
    # If service unavailable, return web_url fallback
    return {
        "status": "ok",
        "api_used": False,
        "event_id": "",
        "html_link": web_url,
        "web_url": web_url,
    }


def delete_calendar_event(event_id: str) -> bool:
    """
    Delete an event from Google Calendar by event_id.
    """
    print(f"\n--- [DEBUG: GOOGLE CALENDAR DELETE] ---", flush=True)
    print(f"Target Event ID: '{event_id}'", flush=True)

    if not event_id:
        print("[DEBUG: GOOGLE CALENDAR DELETE] No API event_id provided (web link fallback record). Skipping API deletion.\n---------------------------------------\n", flush=True)
        return True

    calendar_id = getattr(config, "GOOGLE_CALENDAR_ID", "primary")
    service = get_calendar_service()
    if service:
        try:
            service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            print(f"[DEBUG: GOOGLE CALENDAR DELETE] SUCCESS! Deleted Event ID: '{event_id}'\n---------------------------------------\n", flush=True)
            return True
        except Exception as exc:
            logger.error("[GoogleCalendar] API delete event failed: %s", exc)
            print(f"[DEBUG: GOOGLE CALENDAR DELETE] API Delete Error for event '{event_id}': {exc}\n---------------------------------------\n", flush=True)
            return False

    print("[DEBUG: GOOGLE CALENDAR DELETE] API Client unavailable. Skipped API deletion.\n---------------------------------------\n", flush=True)
    return True
