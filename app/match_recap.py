"""Match recap pipeline for the Manchester United Fan Companion.

Flow:
  run_match_recap_pipeline(user_msg)
        ↓
  fetch_last_match()          — TheSportsDB eventslast → most recent finished match
        ↓
  _check_recap_cache(idEvent) — recap_store lookup; return early on HIT
        ↓
  build_recap_search_query()  — construct NewsAPI q, from_date, to_date
        ↓
  fetch_recap_articles()      — NewsAPI /everything, pageSize=5
        ↓
  ingest_recap_articles()     — scrape + chunk + embed + upsert (source=article)
        ↓
  retrieve_recap_chunks()     — top-5 semantic search on recap_store
        ↓
  generate_match_recap()      — Azure OpenAI → 6-section structured recap
        ↓
  _wipe_raw_chunks()          — delete all docs where source != "llm_recap"
        ↓
  persist_recap()             — chunk + embed + upsert recap (source=llm_recap)
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import unicodedata

import requests
import tiktoken
import trafilatura  # type:ignore

from . import config
from .retrieval import get_recap_collection, get_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
_ENC = tiktoken.get_encoding("cl100k_base")
MAX_CHUNK_TOKENS = 400

# ---------------------------------------------------------------------------
# TheSportsDB — Man Utd team ID will be fetched dynamically via searchteams.php
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# STEP 1 — Fetch all recent finished Man Utd matches from TheSportsDB
# ---------------------------------------------------------------------------
def fetch_recent_finished_matches() -> list:
    """
    Call TheSportsDB to fetch recent matches. Uses searchteams.php to get the team ID dynamically.

    Returns ALL events in the response that have strStatus == 'FT' or 'Match Finished',
    ordered most-recent-first (API default).

    Raises RuntimeError if the API returns no events at all.
    """
    api_key = (config.SPORTS_DB_API_KEY or "").strip()
    if not api_key:
        api_key = "3"

    base_url = f"{config.SPORTS_DB_BASE}/{api_key}"
    team_name = "Manchester United"

    print(f"\n--- [DEBUG: RECAP - TheSportsDB] ---", flush=True)

    # 1. Get team ID dynamically
    try:
        resp = requests.get(f"{base_url}/searchteams.php", params={"t": team_name}, timeout=15)
        resp.raise_for_status()
        teams = resp.json().get("teams")
        if not teams:
            raise ValueError(f"Team '{team_name}' not found")
        team_id = teams[0]["idTeam"]
        print(f"[DEBUG: RECAP] Found team ID for {team_name}: {team_id}", flush=True)
    except Exception as e:
        logger.error(f"[RECAP] Failed to get team ID: {e}")
        # fallback to known ID if search fails
        team_id = "133612"
        print(f"[DEBUG: RECAP] Fallback to known team ID: {team_id}", flush=True)

    # 2. Get last events
    print(f"GET {base_url}/eventslast.php?id={team_id}", flush=True)
    resp = requests.get(f"{base_url}/eventslast.php", params={"id": team_id}, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results") or []

    if not results:
        raise RuntimeError(
            "TheSportsDB returned no recent events for Manchester United."
        )

    # Collect all finished matches
    finished = [
        e for e in results
        if (e.get("strStatus") or "").strip() in ("Match Finished", "FT")
    ]

    if finished:
        print(
            f"[DEBUG: RECAP] Found {len(finished)} finished match(es):",
            flush=True,
        )
        for e in finished:
            print(
                f"  - {e.get('strEvent')} on {e.get('dateEvent')}",
                flush=True,
            )
    else:
        # Fallback: treat every returned event as a candidate
        logger.warning(
            "No events with strStatus='Match Finished' or 'FT' — using all returned results."
        )
        finished = results
        print(
            f"[DEBUG: RECAP] Fallback: using all {len(finished)} result(s).",
            flush=True,
        )

    print("------------------------------------\n", flush=True)
    return finished


# ---------------------------------------------------------------------------
# STEP 2 — Cache check
# ---------------------------------------------------------------------------
def _check_recap_cache(id_event: str) -> Optional[str]:
    """
    Query recap_store for an existing LLM recap for this match.

    Returns the cached recap text on HIT, or None on MISS.
    """
    collection = get_recap_collection()
    if collection.count() == 0:
        return None

    results = collection.get(
        where={"$and": [{"match_id": {"$eq": id_event}}, {"source": {"$eq": "llm_recap"}}]},
        include=["documents"],
    )
    docs = results.get("documents") or []
    if docs:
        logger.info("[RECAP] Cache HIT for match_id=%s — returning stored recap.", id_event)
        print(f"[DEBUG: RECAP] Cache HIT for match_id={id_event}", flush=True)
        # Re-join all chunks in order (they were stored as individual chunks)
        return "\n\n".join(docs)

    logger.info("[RECAP] Cache MISS for match_id=%s — running full pipeline.", id_event)
    print(f"[DEBUG: RECAP] Cache MISS for match_id={id_event} — running full pipeline.", flush=True)
    return None


def clear_recap_cache() -> bool:
    """Wipe all cached match recaps from ChromaDB recap_store collection."""
    try:
        from .retrieval import get_chroma_client, get_recap_collection
        client = get_chroma_client()
        client.delete_collection("recap_store")
        get_recap_collection()
        print("[DEBUG: RECAP] Wiped all cached recaps from ChromaDB recap_store.", flush=True)
        return True
    except Exception as exc:
        logger.error("[RECAP] Failed to clear recap cache: %s", exc)
        return False


# ---------------------------------------------------------------------------
# STEP 3 — Build NewsAPI search query
# ---------------------------------------------------------------------------
def build_recap_search_query(match: dict) -> dict:
    """
    Build the NewsAPI /everything query parameters for a given match.

    Uses the match name (e.g. "Manchester United vs Arsenal") as the core
    search term, scoped to a 4-day window around the match date.
    """
    def clean_name(name: str) -> str:
        return "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

    home = clean_name(match.get("strHomeTeam", "Manchester United"))
    away = clean_name(match.get("strAwayTeam", "Unknown"))
    date_str = match.get("dateEvent", "")  # e.g. "2026-08-03"

    # Build search window: from match day - 7 to match day + 7
    try:
        match_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        match_date = datetime.now(timezone.utc) - timedelta(days=7)

    from_date = (match_date - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = (match_date + timedelta(days=7)).strftime("%Y-%m-%d")

    # Ensure today doesn't exceed NewsAPI window (free tier: last 30 days)
    today = datetime.now(timezone.utc)
    if match_date < today - timedelta(days=29):
        # Match is too old for NewsAPI free tier — widen to last 30 days
        from_date = (today - timedelta(days=29)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

    # Identify the opponent
    opponent = away if "manchester" in home.lower() or "man utd" in home.lower() else home

    # Relaxed query: just require Man Utd aliases + the opponent's name.
    # We drop the strict (recap OR "match report") since the narrow 14-day 
    # window and sports-only domains naturally surface match coverage.
    query = f'("Manchester United" OR "Man Utd" OR "MUFC") AND "{opponent}"'

    print(f"[DEBUG: RECAP] NewsAPI query: {query}", flush=True)
    print(f"[DEBUG: RECAP] Date range: {from_date} to {to_date}", flush=True)

    return {
        "q": query,
        "from": from_date,
        "to": to_date,
        "domains": ",".join(config.RECAP_DOMAINS),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10,
    }


# ---------------------------------------------------------------------------
# STEP 4 — Fetch articles from NewsAPI
# ---------------------------------------------------------------------------
def fetch_recap_articles(query_params: dict) -> list:
    """
    Hit NewsAPI /everything with the given params.

    Returns the list of article dicts. Empty list on error (so pipeline
    continues gracefully even if NewsAPI fails).
    """
    api_key = os.environ.get("NEWS_API_KEY") or config.NEWS_API_KEY
    if not api_key:
        logger.warning("[RECAP] NEWS_API_KEY not set — skipping article fetch.")
        return []

    params = {**query_params, "apiKey": api_key}

    print("\n--- [DEBUG: RECAP - NewsAPI] ---", flush=True)
    try:
        resp = requests.get(
            f"{config.NEWS_API_BASE}/everything",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.warning("[RECAP] NewsAPI error: %s", data.get("message"))
            return []

        articles = data.get("articles", [])
        print(f"[DEBUG: RECAP] NewsAPI returned {len(articles)} article(s).", flush=True)
        for i, a in enumerate(articles, 1):
            print(f"  {i}. {a.get('title')} ({a.get('source', {}).get('name')})", flush=True)
        print("-------------------------------\n", flush=True)
        return articles

    except Exception as exc:
        logger.warning("[RECAP] NewsAPI fetch failed: %s", exc)
        print(f"[DEBUG: RECAP] NewsAPI fetch error: {exc}", flush=True)
        print("-------------------------------\n", flush=True)
        return []


# ---------------------------------------------------------------------------
# Shared helpers — scraping and chunking
# ---------------------------------------------------------------------------
def _scrape_body(article: dict) -> str:
    """Scrape full article text via trafilatura; fall back to NewsAPI snippet."""
    url = article.get("url", "")
    body = ""
    if url:
        try:
            downloaded = trafilatura.fetch_url(url) #type:ignore
            if downloaded:
                body = (
                    trafilatura.extract( # type:ignore
                        downloaded,
                        include_comments=False,
                        include_tables=False,
                        no_fallback=False,
                    )
                    or ""
                )
        except Exception as exc:
            logger.warning("[RECAP] trafilatura failed for %s: %s", url, exc)

    if not body:
        body = article.get("content") or article.get("description") or ""

    return body.strip()


def _chunk_text(text: str, base_metadata: dict) -> list:
    """
    Split text into paragraphs fitting MAX_CHUNK_TOKENS.
    Returns list of (chunk_text, chunk_metadata) tuples.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_tokens = 0
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx
        if not current:
            return
        txt = "\n".join(current)
        meta = {**base_metadata, "chunk_index": chunk_idx}
        chunks.append((txt, meta))
        chunk_idx += 1

    for para in paragraphs:
        para_tokens = len(_ENC.encode(para))
        if current_tokens + para_tokens > MAX_CHUNK_TOKENS and current:
            flush()
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    flush()
    return chunks


# ---------------------------------------------------------------------------
# STEP 4b — Ingest article chunks into recap_store
# ---------------------------------------------------------------------------
def ingest_recap_articles(articles: list, match_id: str) -> int:
    """
    For each article: scrape -> chunk -> embed -> upsert into recap_store.
    Tags every chunk with source="article" and match_id.

    Returns total number of chunks ingested.
    """
    collection = get_recap_collection()
    total_chunks = 0

    print(f"\n--- [DEBUG: RECAP - Ingest] Ingesting {len(articles)} article(s) ---", flush=True)

    for idx, article in enumerate(articles, 1):
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        source_name = (article.get("source") or {}).get("name", "Unknown")

        print(f"  [{idx}] '{title}' ({source_name})", flush=True)

        body = _scrape_body(article)
        if not body:
            print(f"       Skipped (empty body)", flush=True)
            continue

        article_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        base_meta = {
            "source": "article",
            "match_id": match_id,
            "article_id": article_id,
            "title": title,
            "outlet": source_name,
            "url": url,
            "published_at": article.get("publishedAt", ""),
        }

        chunks = _chunk_text(body, base_meta)
        if not chunks:
            print(f"       Skipped (no chunks)", flush=True)
            continue

        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        ids = [
            f"recap-{match_id}-{article_id}-chunk-{i}"
            for i in range(len(chunks))
        ]

        embeddings = [get_embedding(t) for t in texts]
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)

        total_chunks += len(chunks)
        print(f"       Upserted {len(chunks)} chunk(s)", flush=True)

    print(f"[DEBUG: RECAP - Ingest] Total chunks ingested: {total_chunks}", flush=True)
    print("-------------------------------------------------------\n", flush=True)
    return total_chunks


# ---------------------------------------------------------------------------
# STEP 5 — Retrieve top-k chunks from recap_store
# ---------------------------------------------------------------------------
def retrieve_recap_chunks(query: str, top_k: int = 5) -> list:
    """
    Semantic search on recap_store.
    Returns list of dicts: {text, title, outlet, url, published_at, distance}.
    """
    collection = get_recap_collection()
    count = collection.count()
    if count == 0:
        return []

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        chunks.append(
            {
                "text": doc,
                "title": meta.get("title", ""),
                "outlet": meta.get("outlet", meta.get("source", "")),
                "url": meta.get("url", ""),
                "published_at": meta.get("published_at", ""),
                "distance": dist,
            }
        )

    chunks.sort(key=lambda x: x["distance"])
    return chunks


# ---------------------------------------------------------------------------
# STEP 6 — Generate the structured recap via Azure OpenAI
# ---------------------------------------------------------------------------
_RECAP_SYSTEM_PROMPT = (
    "You are the Man Utd Fan Companion. Your task is to produce a structured, grounded match recap "
    "using ONLY the match metadata and article context provided below. "
    "Do not use your general knowledge to fill gaps — if information is missing, write "
    "'Not available in coverage.' for that section."
)

_RECAP_USER_TEMPLATE = """\
[Match Metadata]
Match      : {strEvent}
Date       : {dateEvent}
Competition: {strLeague}
Home Team  : {strHomeTeam}
Away Team  : {strAwayTeam}
Score      : {intHomeScore}-{intAwayScore} ({result})
Venue      : {strVenue}

[Retrieved Article Context]
{formatted_chunks}

[Instructions]
Produce a highly engaging, structured match recap formatted in rich Markdown. Use emojis, bold text, and bulleted lists to make it easy and exciting to read. Include exactly these six sections:

## 🏆 Winning Result
Include the score, competition, venue, and date in a clear, formatted bulleted list.

## 📰 Match Summary
Provide a 2-3 paragraph narrative overview. Use **bolding** for key phrases or player names to make it pop.

## ⚡ Key Moments
Use a bulleted list to highlight goals, cards, and turning points (include the minute if available). Use relevant emojis like ⚽ for goals or 🟨/🟥 for cards.

## ♟️ Tactical Analysis
Provide a breakdown of formations, pressing shape, and key phases of play. Use bullet points or short paragraphs.

## ⭐ Standout Performers
List the top 2-3 players. Use bullet points and provide a brief justification for each.

## 🔗 Sources
Cite the article title + outlet for every factual claim used. Format as a bulleted list.

Never invent facts. If a section has no supporting information, write "*Not available in coverage.*"
"""


def _determine_result(match: dict) -> str:
    """Return a human-readable result string, e.g. 'Manchester United won' or 'Draw'."""
    home = match.get("strHomeTeam", "Home")
    away = match.get("strAwayTeam", "Away")
    try:
        hs = int(match.get("intHomeScore") or 0)
        as_ = int(match.get("intAwayScore") or 0)
    except (ValueError, TypeError):
        return "Score not available"

    if hs > as_:
        return f"{home} won"
    if as_ > hs:
        return f"{away} won"
    return "Draw"


def _format_chunks_for_prompt(chunks: list) -> str:
    """Format retrieved chunks into a readable block for the LLM prompt."""
    if not chunks:
        return "No article context available."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        pub = chunk["published_at"][:10] if chunk["published_at"] else "unknown date"
        lines.append(f'\n--- Source {i}: "{chunk["title"]}" ({chunk["outlet"]}, {pub}) ---')
        if chunk["url"]:
            lines.append(f"URL: {chunk['url']}")
        lines.append(chunk["text"])

    return "\n".join(lines)


def generate_match_recap(match: dict, chunks: list) -> str:
    """
    Call Anthropic MODEL_SONNET to generate a 6-section match recap.

    Returns the raw recap text string.
    """
    from .main import _client

    formatted_chunks = _format_chunks_for_prompt(chunks)
    result_str = _determine_result(match)

    user_content = _RECAP_USER_TEMPLATE.format(
        strEvent=match.get("strEvent", "Manchester United vs Opponent"),
        dateEvent=match.get("dateEvent", "Unknown"),
        strLeague=match.get("strLeague", "Unknown Competition"),
        strHomeTeam=match.get("strHomeTeam", "Home"),
        strAwayTeam=match.get("strAwayTeam", "Away"),
        intHomeScore=match.get("intHomeScore", "?"),
        intAwayScore=match.get("intAwayScore", "?"),
        result=result_str,
        strVenue=match.get("strVenue", "Unknown Venue"),
        formatted_chunks=formatted_chunks,
    )

    print("\n--- [DEBUG: RECAP - LLM Generation] ---", flush=True)
    print(f"Sending prompt to {config.MODEL_SONNET}...", flush=True)

    resp = _client().messages.create(
        model=config.MODEL_SONNET,
        temperature=0.3,
        max_tokens=2048,
        system=_RECAP_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_content},
        ],
    )

    recap_text = resp.content[0].text
    print(f"[DEBUG: RECAP] Recap generated ({len(recap_text)} chars)", flush=True)
    print("---------------------------------------\n", flush=True)
    return recap_text


# ---------------------------------------------------------------------------
# STEP 7 — Wipe raw article chunks from recap_store
# ---------------------------------------------------------------------------
def _wipe_raw_chunks() -> None:
    """
    Delete all documents in recap_store where source != 'llm_recap'.
    This clears article chunks after the LLM recap has been generated,
    keeping the store clean for the next request.
    """
    collection = get_recap_collection()
    if collection.count() == 0:
        return

    results = collection.get(
        where={"source": {"$ne": "llm_recap"}},
        include=[],
    )
    ids_to_delete = results.get("ids") or []
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("[RECAP] Wiped %d raw article chunk(s) from recap_store.", len(ids_to_delete))
        print(f"[DEBUG: RECAP] Wiped {len(ids_to_delete)} raw chunk(s).", flush=True)


# ---------------------------------------------------------------------------
# STEP 8 — Persist the LLM recap into recap_store
# ---------------------------------------------------------------------------
def persist_recap(recap_text: str, match: dict) -> None:
    """
    Chunk the LLM recap text, embed each chunk, and upsert into recap_store
    with source='llm_recap' and match_id metadata.

    This persists the recap so future requests for the same match
    get an instant cache hit without re-running the pipeline.
    """
    match_id = match.get("idEvent", "unknown")
    match_name = match.get("strEvent", "Unknown Match")
    generated_at = datetime.now(timezone.utc).isoformat()

    base_meta = {
        "source": "llm_recap",
        "match_id": match_id,
        "match_name": match_name,
        "generated_at": generated_at,
    }

    chunks = _chunk_text(recap_text, base_meta)
    if not chunks:
        logger.warning("[RECAP] persist_recap: no chunks produced from recap text.")
        return

    collection = get_recap_collection()
    texts = [c[0] for c in chunks]
    metas = [c[1] for c in chunks]
    ids = [f"recap-llm-{match_id}-chunk-{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(t) for t in texts]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)

    logger.info(
        "[RECAP] Persisted LLM recap for '%s' (%d chunks).", match_name, len(chunks)
    )
    print(
        f"[DEBUG: RECAP] Persisted LLM recap: {len(chunks)} chunk(s) for match_id={match_id}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Public entrypoint — orchestrates all 8 steps
# ---------------------------------------------------------------------------
def _build_disambiguation_message(matches: list) -> str:
    """
    Build a friendly message listing all candidate matches for the user to
    choose from when more than one finished match is found.
    """
    lines = [
        "I found **multiple recent Manchester United matches** — which one would you like a recap of?\n"
    ]
    for i, m in enumerate(matches, 1):
        home = m.get("strHomeTeam", "?")
        away = m.get("strAwayTeam", "?")
        date = m.get("dateEvent", "unknown date")
        league = m.get("strLeague", "")
        hs = m.get("intHomeScore", "?")
        as_ = m.get("intAwayScore", "?")
        score_str = f"  ({hs}–{as_})" if hs != "?" else ""
        league_str = f" | {league}" if league else ""
        lines.append(f"{i}. **{home} vs {away}** — {date}{score_str}{league_str}")

    lines.append(
        "\nJust tell me which match, e.g. *'recap the Arsenal game'* or the date."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entrypoint — orchestrates all steps
# ---------------------------------------------------------------------------
def run_match_recap_pipeline(user_msg: str) -> str:
    """
    Orchestrate the full match recap pipeline.

    Steps:
      1. Fetch all recent finished Man Utd matches from TheSportsDB
      1a. If >1 match found, return a disambiguation message asking the user
          to specify which match they want.
      2. Check recap_store cache — return immediately on HIT
      3. Build NewsAPI query from match metadata (±7-day window)
      4. Fetch recap articles from NewsAPI
      5. Ingest articles into recap_store (source=article)
      6. Retrieve top-5 chunks for the user's message
      7. Generate structured LLM recap
      8. Wipe raw article chunks from recap_store
      9. Persist LLM recap into recap_store

    Returns the generated (or cached) recap text, or a disambiguation prompt
    if multiple matches are found.
    Falls back to a friendly error message if a critical step fails.
    """
    print("\n========================================", flush=True)
    print("[DEBUG: RECAP PIPELINE] Starting...", flush=True)
    print("========================================", flush=True)

    try:
        # STEP 1 — Get all recent finished matches
        matches = fetch_recent_finished_matches()

        # STEP 1a — Disambiguate if more than one match found
        if len(matches) > 1:
            print(
                f"[DEBUG: RECAP PIPELINE] {len(matches)} matches found — returning disambiguation.",
                flush=True,
            )
            return _build_disambiguation_message(matches)

        if len(matches) == 0:
            return (
                "I couldn't find any recent finished matches for Manchester United. "
                "Please try again later."
            )

        match = matches[0]
        match_id = match.get("idEvent", "unknown")

        # STEP 2 — Cache check
        cached = _check_recap_cache(match_id)
        if cached:
            print("[DEBUG: RECAP PIPELINE] Returning cached recap.\n", flush=True)
            return cached

        # STEP 3 — Build query (±7-day window)
        query_params = build_recap_search_query(match)

        # STEP 4 — Fetch articles
        articles = fetch_recap_articles(query_params)

        # STEP 5 — Ingest articles (even if empty — pipeline continues)
        if articles:
            ingest_recap_articles(articles, match_id)
        else:
            logger.warning(
                "[RECAP] No articles found — recap will rely on match metadata only."
            )
            print(
                "[DEBUG: RECAP] No articles found — will rely on match metadata only.",
                flush=True,
            )

        # STEP 6 — Retrieve relevant chunks
        retrieval_query = (
            f"{match.get('strEvent', 'Manchester United match')} recap match report"
        )
        chunks = retrieve_recap_chunks(retrieval_query, top_k=5)
        print(
            f"[DEBUG: RECAP] Retrieved {len(chunks)} chunk(s) for LLM context.",
            flush=True,
        )

        # STEP 7 — Generate recap
        recap_text = generate_match_recap(match, chunks)

        # STEP 8 — Wipe raw chunks
        _wipe_raw_chunks()

        # STEP 9 — Persist LLM recap
        persist_recap(recap_text, match)

        print("[DEBUG: RECAP PIPELINE] Complete.\n", flush=True)
        return recap_text

    except Exception as exc:
        logger.error("[RECAP PIPELINE] Failed: %s", exc, exc_info=True)
        print(f"[DEBUG: RECAP PIPELINE] ERROR: {exc}", flush=True)
        return (
            "Sorry, I wasn't able to generate a match recap right now. "
            "The service may be temporarily unavailable — please try again in a moment."
        )
