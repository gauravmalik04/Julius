"""Configuration — environment variables and paths.

Add any new environment variables your agent needs here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"       # knowledge documents live here
CHROMA_DIR = ROOT / ".chroma"  # persistent ChromaDB vector store

# ---------------------------------------------------------------------------
# Anthropic Models
# ---------------------------------------------------------------------------
_raw_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or os.getenv("CLAUDE_API_KEY")
_raw_base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_URL") or os.getenv("CLAUDE_BASE_URL")

ANTHROPIC_API_KEY = _raw_api_key.strip() if _raw_api_key and _raw_api_key.strip() else None
_clean_base_url = _raw_base_url.strip() if _raw_base_url and _raw_base_url.strip() else None

if "ANTHROPIC_BASE_URL" in os.environ and not os.environ["ANTHROPIC_BASE_URL"].strip():
    del os.environ["ANTHROPIC_BASE_URL"]

ANTHROPIC_BASE_URL = _clean_base_url
MODEL_SONNET = os.getenv("MODEL_SONNET", "claude-sonnet-4-6")   # Higher thinking/reasoning tasks
MODEL_HAIKU = os.getenv("MODEL_HAIKU", "claude-haiku-4-5")     # Small & simpler classification tasks

# Azure OpenAI (Legacy)
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# External integrations (Phase 3+)
# ---------------------------------------------------------------------------
# Football-Data.org (football-api) — fixtures, standings, squad roster
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")          # Football-Data.org
FOOTBALL_API_BASE = "https://api.football-data.org/v4"

# API-Football / API-Sports (api-football) — injuries endpoint
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")            # API-Football (v3.football.api-sports.io)
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")                   # NewsAPI.org
NEWS_API_BASE = "https://newsapi.org/v2"

# TheSportsDB — used by the match recap pipeline
SPORTS_DB_API_KEY = os.getenv("SPORTS_DB_API_KEY", "3")        # free public key = "3"
SPORTS_DB_BASE = "https://www.thesportsdb.com/api/v1/json"

# Manchester United team ID on football-data.org (v4)
MAN_UTD_TEAM_ID = 66

# Manchester United team ID on API-Football and TheSportsDB
MAN_UTD_API_FOOTBALL_ID = 33
MAN_UTD_SPORTS_DB_ID = 33

# News ingestion settings
NEWS_QUERY = '"Manchester United" OR "Man Utd" OR "MUFC"'
NEWS_INGEST_DAYS_BACK = 7       # How far back to look on each ingest run
NEWS_MAX_ARTICLES = 50          # Max articles per ingest call
ALLOWED_DOMAINS = [
    "manutd.com",               # Official club
    "skysports.com",
    "bbc.com",
    "theathletic.com",
    "reuters.com",
    "stadiumrant.com",
]

# Domains used specifically for match recap article scraping
RECAP_DOMAINS = [
    "bbc.com",
    "theguardian.com",
    "espn.com",
    "skysports.com",
    "theathletic.com",
    "manutd.com",
    "stretfordpaddock.com",
    "manchestereveningnews.co.uk",
    "stadiumrant.com",
]

GOOGLE_CALENDAR_CREDS = os.getenv(                             # Path to credentials.json
    "GOOGLE_CALENDAR_CREDS_PATH", str(ROOT / "credentials.json")
)
