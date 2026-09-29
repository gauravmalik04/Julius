"""News ingestion pipeline for the Manchester United Fan Companion.

Flow:
  fetch_articles()  — NewsAPI /everything, domain-filtered
        ↓
  scrape_body()     — trafilatura pulls full article text; falls back to NewsAPI snippet
        ↓
  chunk_article()   — splits body into ≤400-token paragraphs
        ↓
  ingest()          — dedup via sha256(url) ID, embed, upsert to ChromaDB news_store
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import tiktoken
# pyrefly: ignore [missing-import]
import trafilatura

from . import config
from .retrieval import get_news_collection, get_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokeniser — used for chunk size control
# ---------------------------------------------------------------------------
_ENC = tiktoken.get_encoding("cl100k_base")

MAX_CHUNK_TOKENS = 400

# ---------------------------------------------------------------------------
# Ingest log — tracks last run timestamp to avoid redundant fetches
# ---------------------------------------------------------------------------
_INGEST_LOG = config.DATA_DIR / "ingest_log.json"


def _load_ingest_log() -> dict:
    if _INGEST_LOG.exists():
        return json.loads(_INGEST_LOG.read_text())
    return {}


def _save_ingest_log(log: dict) -> None:
    _INGEST_LOG.write_text(json.dumps(log, indent=2))


# ---------------------------------------------------------------------------
# Step 1 — Fetch article metadata from NewsAPI
# ---------------------------------------------------------------------------
def fetch_articles(days_back: int = config.NEWS_INGEST_DAYS_BACK) -> list[dict]:
    """
    Query NewsAPI /everything for Manchester United articles from approved domains.
    Returns a list of raw article dicts from the API response.
    """
    import os
    api_key = os.environ.get("NEWS_API_KEY") or config.NEWS_API_KEY
    if not api_key:
        raise RuntimeError(
            "NEWS_API_KEY is not set. Add it to your .env file."
        )

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
        "%Y-%m-%d"
    )

    print(f"\n--- [DEBUG: NEWS INGESTION - FETCH] ---", flush=True)
    print(f"Querying NewsAPI from_date={from_date}, days_back={days_back}", flush=True)

    params = {
        "q": config.NEWS_QUERY,
        "domains": ",".join(config.ALLOWED_DOMAINS),
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": config.NEWS_MAX_ARTICLES,
        "apiKey": api_key,
    }

    resp = requests.get(
        f"{config.NEWS_API_BASE}/everything",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", [])
    logger.info("NewsAPI returned %d articles", len(articles))
    print(f"NewsAPI returned {len(articles)} total articles.", flush=True)
    print("-----------------------------------------\n", flush=True)
    return articles


# ---------------------------------------------------------------------------
# Step 2 — Scrape full article body
# ---------------------------------------------------------------------------
def scrape_body(article: dict) -> str:
    """
    Use trafilatura to scrape the full article body from the URL.
    Falls back to NewsAPI's own content/description snippet on failure.
    """
    url = article.get("url", "")
    body = ""

    if url:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                body = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                ) or ""
        except Exception as exc:
            logger.warning("trafilatura failed for %s: %s", url, exc)

    # Fallback: use NewsAPI content snippet (usually truncated at 200 chars)
    if not body:
        body = article.get("content") or article.get("description") or ""
        if body:
            logger.debug("Using NewsAPI snippet fallback for %s", url)

    return body.strip()


# ---------------------------------------------------------------------------
# Step 3 — Build stable article ID
# ---------------------------------------------------------------------------
def build_article_id(article: dict) -> str:
    """
    Deterministic dedup key: sha256 of the article URL.
    Same article ingested twice → same ID → ChromaDB upsert is a no-op.
    """
    url = article.get("url", article.get("title", ""))
    return hashlib.sha256(url.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Step 4 — Chunk body into ≤MAX_CHUNK_TOKENS token paragraphs
# ---------------------------------------------------------------------------
def chunk_article(body: str, metadata: dict) -> list[tuple[str, dict]]:
    """
    Split article body into paragraph-level chunks that fit within
    MAX_CHUNK_TOKENS. Returns list of (chunk_text, chunk_metadata) tuples.
    """
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    chunks: list[tuple[str, dict]] = []
    current: list[str] = []
    current_tokens = 0

    def flush(idx: int) -> None:
        if not current:
            return
        text = "\n".join(current)
        chunk_meta = {
            **metadata,
            "chunk_index": idx,
            "chunk_id": f"{metadata['article_id']}-chunk-{idx}",
        }
        chunks.append((text, chunk_meta))

    chunk_idx = 0
    for para in paragraphs:
        para_tokens = len(_ENC.encode(para))
        if current_tokens + para_tokens > MAX_CHUNK_TOKENS and current:
            flush(chunk_idx)
            chunk_idx += 1
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    flush(chunk_idx)
    return chunks


# ---------------------------------------------------------------------------
# Main entrypoint — orchestrates the full pipeline
# ---------------------------------------------------------------------------
def ingest(days_back: int = config.NEWS_INGEST_DAYS_BACK) -> dict:
    """
    Run the full ingestion pipeline:
      fetch → scrape → dedup → chunk → embed → upsert

    Returns a summary dict: {fetched, new_articles, new_chunks, skipped}
    """
    print(f"\n==================================================", flush=True)
    print(f"[DEBUG: STARTING NEWS INGESTION PIPELINE]", flush=True)
    print(f"==================================================", flush=True)
    collection = get_news_collection()
    articles = fetch_articles(days_back)

    stats = {"fetched": len(articles), "new_articles": 0, "new_chunks": 0, "skipped": 0}

    for idx, article in enumerate(articles, 1):
        article_id = build_article_id(article)
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        print(f"[DEBUG: NEWS INGESTION] Processing article {idx}/{len(articles)}: '{title}'", flush=True)

        # --- Dedup check: has this article already been ingested? ---
        existing = collection.get(
            where={"article_id": {"$eq": article_id}},
            limit=1,
        )
        if existing and existing.get("ids"):
            print(f"  -> Skipped (Already ingested ID: {article_id[:8]}...)", flush=True)
            stats["skipped"] += 1
            continue

        # --- Scrape full body ---
        body = scrape_body(article)
        if not body:
            logger.warning("Empty body for article: %s — skipping", url)
            print(f"  -> Skipped (Empty body for {url})", flush=True)
            stats["skipped"] += 1
            continue

        # --- Build metadata ---
        source_name = (article.get("source") or {}).get("name", "Unknown")
        metadata = {
            "article_id": article_id,
            "title": title,
            "source": source_name,
            "author": article.get("author") or source_name,
            "published_at": article.get("publishedAt", ""),
            "url": url,
            "team": "MUN",
        }

        # --- Chunk ---
        chunks = chunk_article(body, metadata)
        if not chunks:
            print(f"  -> Skipped (No chunks produced)", flush=True)
            stats["skipped"] += 1
            continue

        # --- Embed & upsert ---
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        chunk_ids = [m["chunk_id"] for m in metas]

        print(f"  -> Embedding & upserting {len(chunks)} chunk(s) from '{source_name}'...", flush=True)
        embeddings = [get_embedding(t) for t in texts]

        collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

        stats["new_articles"] += 1
        stats["new_chunks"] += len(chunks)
        logger.info(
            "Ingested: '%s' (%s) — %d chunks",
            metadata["title"],
            source_name,
            len(chunks),
        )
        print(f"  -> Ingested successfully!", flush=True)

    # Save last-run timestamp
    log = _load_ingest_log()
    log["last_ingested_at"] = datetime.now(timezone.utc).isoformat()
    log["last_stats"] = stats
    _save_ingest_log(log)

    logger.info("Ingest complete: %s", stats)
    print(f"\n==================================================", flush=True)
    print(f"[DEBUG: NEWS INGESTION COMPLETE] Final Stats: {stats}", flush=True)
    print(f"==================================================\n", flush=True)
    return stats
