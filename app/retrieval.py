"""Retrieval layer for the Manchester United Fan Companion.

Provides:
  get_chroma_client()   — singleton persistent ChromaDB client
  get_news_collection() — the 'news_store' collection, created if absent
  get_embedding(text)   — embed a string via Azure OpenAI
  retrieve_news(query)  — semantic top-k search, returns chunks + metadata
"""

import logging
from functools import lru_cache
from typing import Optional

import chromadb
# pyrefly: ignore [missing-import]
from openai import AzureOpenAI

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB client — singleton, persisted to .chroma/
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a singleton ChromaDB persistent client stored at CHROMA_DIR."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_news_collection() -> chromadb.Collection:
    """
    Return the 'news_store' ChromaDB collection.
    Created automatically if it does not yet exist.
    We use cosine distance as our similarity metric.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="news_store",
        metadata={"hnsw:space": "cosine"},
    )


def get_recap_collection() -> chromadb.Collection:
    """
    Return the 'recap_store' ChromaDB collection used by the match recap pipeline.
    Created automatically if it does not yet exist.
    Stores both raw article chunks (source=article) and the final LLM recap (source=llm_recap).
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="recap_store",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Embedding helper — SentenceTransformers (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _embed_model():
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> list[float]:
    """
    Embed a single string using local SentenceTransformer model.
    Returns a list of floats (the embedding vector).
    """
    model = _embed_model()
    # SentenceTransformer.encode returns a numpy array, convert to list
    return model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_news(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the user query and run a cosine similarity search against news_store.

    Returns a list of result dicts, each containing:
      - text        : the chunk text
      - title       : article title
      - source      : publication name (e.g. "BBC Sport")
      - author      : article author
      - published_at: ISO timestamp string
      - url         : original article URL
      - distance    : cosine distance (lower = more similar)
    """
    collection = get_news_collection()

    # Check collection is populated
    count = collection.count()
    if count == 0:
        logger.info("news_store is empty — returning no results")
        return []

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": doc,
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "author": meta.get("author", ""),
                "published_at": meta.get("published_at", ""),
                "url": meta.get("url", ""),
                "distance": dist,
            }
        )

    # Sort by distance ascending (most relevant first)
    chunks.sort(key=lambda x: x["distance"])
    return chunks


# ---------------------------------------------------------------------------
# Format retrieved chunks for LLM injection
# ---------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved news chunks into a clean [Context] block
    ready to be prepended to the user's message.
    """
    if not chunks:
        return ""

    lines = ["[Retrieved News Context]"]
    for i, chunk in enumerate(chunks, start=1):
        pub = chunk["published_at"][:10] if chunk["published_at"] else "unknown date"
        lines.append(
            f"\n--- Article {i}: \"{chunk['title']}\" "
            f"({chunk['source']}, {pub}) ---"
        )
        if chunk["url"]:
            lines.append(f"Source URL: {chunk['url']}")
        lines.append(chunk["text"])

    return "\n".join(lines)
