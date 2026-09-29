# JULIUS : Manchester United Fan Companion Agent

A dedicated **Manchester United fan companion** built on FastAPI, Claude (Anthropic), ChromaDB, and LangGraph. The agent answers questions about fixtures, live stats, match recaps, news & transfers, and manages Google Calendar reminders — scoped exclusively to Manchester United.

---

## Architecture

```
User Message
    ↓
LLM Guardrail + Intent Classifier  (claude-haiku-4-5)
    ↓
LangGraph Orchestration Pipeline   (agent_graph.py)
    ├─ scope_guard       → warm redirect if out-of-scope
    ├─ intent_classifier → routes to the right tool
    └─ Conditional Router
         ├─ schedule_and_fixtures_tool  → fixtures_handler.py  (football-data.org API)
         ├─ live_stats_lookup_tool      → stats_handler.py     (football-data.org + API-Football)
         ├─ match_recap_tool            → match_recap.py       (TheSportsDB + NewsAPI + ChromaDB)
         ├─ news_and_transfers_tool     → news_handler.py      (ChromaDB news_store)
         ├─ add_to_calendar_tool        → reminders_handler.py (Google Calendar API)
         └─ general_chitchat_tool       → direct LLM response
    ↓
Answer Generator (claude-sonnet-4-6)
    ↓
Grounded Response + Citations → User
```

---

## Project Structure

```
app/
  main.py              FastAPI app, APScheduler, intent classifier, chat endpoint
  agent_graph.py       LangGraph pipeline with @tool-wrapped handlers
  config.py            Environment variables, paths, API config
  retrieval.py         ChromaDB collections (news_store, recap_store)
  news_ingestor.py     NewsAPI fetch → scrape → chunk → embed → upsert
  news_handler.py      Semantic retrieval from news_store
  match_recap.py       TheSportsDB + article scraping → recap pipeline
  stats_handler.py     Squad sync, live standings, player stats, injuries
  fixtures_handler.py  Upcoming/past fixtures from football-data.org
  reminders_handler.py Google Calendar reminder creation/deletion with confirmation flow
  calendar_service.py  Google Calendar API OAuth client
  memory_manager.py    Per-session sliding-window conversation history
  index.html           Single-file chat UI

data/
  users.json           User profiles (currently: Gaurav Malik / Man Utd)
  records.json         Squad roster + manager seed data (synced weekly)
  chat_history.json    Runtime — per-session conversation history [gitignored]
  reminders.json       Runtime — saved match reminders [gitignored]
  ingest_log.json      Runtime — last news ingest timestamp & stats [gitignored]

evals/
  evaluator.py              LLM-as-judge evaluation runner
  golden_dataset.json       Hand-curated eval cases
  eval_results.json         Latest eval run output [gitignored]
  evaluation_round_results.json  Rolling eval history [gitignored]

static/
  devil.png, logo.png, stadium.jpg   UI image assets

.chroma/               ChromaDB persistent storage (news_store + recap_store) [gitignored]
golden_dataset.json    Root-level eval dataset copy
workflow.png           Architecture diagram
```

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
copy .env.example .env
# Fill in your credentials (see Environment Variables section below)
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key (Claude models) |
| `ANTHROPIC_BASE_URL` | Optional | Custom base URL (e.g. Azure-hosted Claude) |
| `FOOTBALL_API_KEY` | ✅ | [football-data.org](https://www.football-data.org) — fixtures, squad, standings |
| `API_FOOTBALL_KEY` | Optional | [API-Football](https://www.api-football.com) — injuries endpoint |
| `NEWS_API_KEY` | ✅ | [NewsAPI.org](https://newsapi.org) — news ingestion pipeline |
| `SPORTS_DB_API_KEY` | Optional | [TheSportsDB](https://www.thesportsdb.com) — match recap pipeline (defaults to free key `3`) |
| `GOOGLE_CALENDAR_CREDS_PATH` | Optional | Path to `credentials.json` for Google Calendar OAuth |
| `AZURE_OPENAI_ENDPOINT` | Legacy | Azure OpenAI endpoint (used by LangGraph tools via AzureChatOpenAI) |
| `AZURE_OPENAI_API_KEY` | Legacy | Azure OpenAI API key |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Legacy | Chat model deployment name (default: `gpt-4o-mini`) |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | Legacy | Embedding deployment name (default: `text-embedding-3-small`) |

> **Never commit `.env`, `credentials.json`, or `token.json`** — all three are gitignored.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Main chat endpoint. Body: `{"message": "...", "session_id": "..."}` |
| `DELETE` | `/api/chat/history` | Clear conversation history for the current session |
| `POST` | `/api/ingest-news` | Manually trigger news ingestion (also runs on cron: 00:00, 06:00, 12:00, 18:00 IST) |
| `GET` | `/api/fixtures-debug` | Inspect raw fixtures context for a given query |
| `GET` | `/api/stats-debug` | Inspect raw stats context for a given query |
| `GET` | `/api/recap-debug` | Manually trigger the full match recap pipeline |
| `DELETE` | `/api/recap-cache` | Wipe all cached recaps from ChromaDB `recap_store` |
| `GET` | `/api/reminders-debug` | View saved reminders and pending confirmation state |
| `GET` | `/` | Serves the chat UI (`app/index.html`) |

---

## Scheduled Jobs (APScheduler)

| Job | Schedule | What it does |
|---|---|---|
| News Ingestion | Every day at 00:00, 06:00, 12:00, 18:00 IST | Fetches latest Man Utd articles from NewsAPI, scrapes full text, chunks, embeds, and upserts into ChromaDB `news_store` |
| Squad Sync | Every Sunday at 00:00 IST | Syncs squad roster and manager from football-data.org into `data/records.json` |

---

## Intent Lanes

The classifier routes every message into one of these lanes:

| Intent | Handler | Data Source |
|---|---|---|
| `fixtures` | `fixtures_handler.py` | football-data.org v4 |
| `stats` | `stats_handler.py` | football-data.org + API-Football + `records.json` |
| `recap` | `match_recap.py` | TheSportsDB + NewsAPI article scraping + ChromaDB |
| `news` | `news_handler.py` | ChromaDB `news_store` (populated by ingestion pipeline) |
| `calendar` | `reminders_handler.py` | Google Calendar API + `data/reminders.json` |
| `general` | Direct LLM | claude-sonnet-4-6 |
| `ambiguous` | Clarification loop | Asks user to disambiguate before routing |
| `out_of_scope` | Warm redirect | No LLM call; canned redirect message |

---

## Google Calendar Setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com).
2. Enable the **Google Calendar API**.
3. Create an **OAuth 2.0 Desktop App** credential and download `credentials.json` to the repo root.
4. On first use, the app opens a browser for OAuth consent and saves `token.json`.

Both files are gitignored — never commit them.

---

## Evaluation

Run the LLM-as-judge evaluator against the golden dataset:

```bash
python -m evals.evaluator
```

Results are written to `evals/eval_results.json` and appended to `evals/evaluation_round_results.json`.
