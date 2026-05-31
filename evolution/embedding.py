"""Text embedding wrapper for ReasoningBank-style cosine retrieval.

We use OpenAI `text-embedding-3-small` (1536-dim, $0.02 / 1M tokens — our
memory bank is tiny so cost is negligible). All vectors are cached to
`runs/.embedding_cache.json` keyed by the sha1 of the source text so
that re-runs don't re-embed.

If the OpenAI SDK is unavailable or no API key is set, `embed()` returns
None and callers degrade gracefully (cosine mode returns empty results).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


_CACHE_PATH = Path(__file__).resolve().parent.parent / "runs" / ".embedding_cache.json"
_CACHE: dict[str, list[float]] | None = None
_CLIENT = None


def _hash(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if _CACHE_PATH.exists():
        try:
            _CACHE = json.loads(_CACHE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            _CACHE = {}
    else:
        _CACHE = {}
    return _CACHE


def _flush_cache() -> None:
    if _CACHE is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False))


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        from openai import OpenAI
    except ImportError:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    _CLIENT = OpenAI()
    return _CLIENT


def embed(text: str, model: str = "text-embedding-3-small") -> Optional[list[float]]:
    """Return the embedding vector for `text`, or None if unavailable.

    Cached on disk by (model, text) sha1."""
    if not text:
        return None
    cache = _load_cache()
    key = _hash(text, model)
    if key in cache:
        return cache[key]
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=model, input=text)
    except Exception:
        return None
    vec = list(resp.data[0].embedding)
    cache[key] = vec
    _flush_cache()
    return vec


def embed_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[Optional[list[float]]]:
    """Vectorise a batch. Cached items skipped; uncached sent in one API call."""
    cache = _load_cache()
    keys = [_hash(t, model) for t in texts]
    out: list[Optional[list[float]]] = [cache.get(k) for k in keys]
    missing_idx = [i for i, v in enumerate(out) if v is None and texts[i]]
    if not missing_idx:
        return out
    client = _get_client()
    if client is None:
        return out  # leave None entries
    try:
        resp = client.embeddings.create(
            model=model,
            input=[texts[i] for i in missing_idx],
        )
        for j, idx in enumerate(missing_idx):
            vec = list(resp.data[j].embedding)
            cache[keys[idx]] = vec
            out[idx] = vec
        _flush_cache()
    except Exception:
        pass
    return out
