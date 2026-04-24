"""
RAG (Retrieval-Augmented Generation) engine for Interexy's case study knowledge base.

How it works:
1. Case studies are stored as Markdown files in knowledge_base/cases/
2. Each file is embedded once using text-embedding-3-small and cached locally
3. Cache uses MD5 hashes of file contents — auto-invalidates when a case is edited
4. At query time: embed the query, compute cosine similarity, return top-K cases

No extra dependencies — cosine similarity is computed with the math module.
"""
import json
import logging
import math
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────── Paths ───────────────────────────────────────────

_BASE_DIR = Path(__file__).parent / "knowledge_base"
_CASES_DIR = _BASE_DIR / "cases"
_CACHE_FILE = _BASE_DIR / "embeddings_cache.json"

# ─────────────────────────── Cosine similarity ───────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────── Case loading ────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML-style frontmatter from a markdown file.

    Expects the file to start with `---`, followed by key: value lines
    (including `  - list items`), and closed by another `---`.

    Returns (metadata_dict, body_text). If no frontmatter found, returns
    ({}, original content).
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    meta: Dict[str, Any] = {}
    current_key: Optional[str] = None
    end_idx = len(lines)

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break
        if line.startswith("  - ") or line.startswith("- "):
            # List item under current_key
            item = line.strip().lstrip("- ").strip()
            if current_key and isinstance(meta.get(current_key), list):
                meta[current_key].append(item)
        elif ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val:
                meta[key] = val
            else:
                # Empty value → list follows
                meta[key] = []
                current_key = key
        # Ignore other lines (blank, nested, etc.)

    body = "\n".join(lines[end_idx + 1:]).strip()
    return meta, body


def load_cases() -> List[Dict[str, Any]]:
    """Load all .md case files from knowledge_base/cases/.

    Parses YAML frontmatter (industry, tech_stack, client_type, region) and
    extracts the title from the first # heading.

    Returns a list of dicts with keys:
      id, title, content, path, hash, industry, tech_stack, client_type, region
    """
    if not _CASES_DIR.exists():
        logger.warning(f"Cases directory not found: {_CASES_DIR}")
        return []

    cases = []
    for md_file in sorted(_CASES_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        meta, body = _parse_frontmatter(content)

        # Title: first # heading in body, fallback to filename
        title = md_file.stem.replace("_", " ").title()
        for line in body.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        cases.append({
            "id": md_file.stem,
            "title": title,
            "content": content,   # full content (frontmatter + body) for embedding
            "body": body,         # body only (for display)
            "path": str(md_file),
            "hash": _md5(content),
            # Metadata from frontmatter
            "industry": meta.get("industry", []) if isinstance(meta.get("industry"), list) else [meta.get("industry", "")],
            "tech_stack": meta.get("tech_stack", []) if isinstance(meta.get("tech_stack"), list) else [meta.get("tech_stack", "")],
            "client_type": meta.get("client_type", ""),
            "region": meta.get("region", ""),
        })

    logger.info(f"Loaded {len(cases)} case(s) from knowledge base")
    return cases


# ─────────────────────────── Embedding & cache ───────────────────────────────

def _load_cache() -> Dict[str, Any]:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupted embeddings cache — will regenerate")
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _make_client(timeout: int = 30) -> httpx.AsyncClient:
    """Create httpx client with proxy support (mirrors server.py _make_client)."""
    proxy_url = os.environ.get("PROXY_URL")
    if proxy_url:
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(proxy=proxy_url),
            timeout=timeout,
        )
    return httpx.AsyncClient(timeout=timeout)


async def _embed_text(text: str, api_key: str) -> List[float]:
    """Call OpenAI text-embedding-3-small and return the embedding vector."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}

    async with _make_client(timeout=30) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def _normalize_query(query: str, api_key: str) -> str:
    """Expand a query into rich English keywords for better embedding recall.

    Short or keyword-only queries ("IoT Bluetooth", "AI", "fintech") produce
    sparse vectors that miss relevant cases even when the cases clearly cover
    that topic. This function uses gpt-4o-mini to expand every query — both
    non-English (translate) and English (enrich with synonyms / related terms)
    — so retrieval quality is consistent regardless of query language or length.

    Returns the original query unchanged if the API call fails.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 80,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Expand the user's query into rich English keywords and synonyms "
                    "suitable for semantic search over a software project portfolio. "
                    "If the query is not in English, translate it first. "
                    "Include related technologies, industry terms, and use-case synonyms. "
                    "Output only the expanded English keywords, nothing else. "
                    "Example: 'IoT Bluetooth' → "
                    "'IoT Internet of Things Bluetooth BLE wireless sensors hardware "
                    "device connectivity embedded smart devices wearables'"
                ),
            },
            {"role": "user", "content": query},
        ],
    }

    try:
        async with _make_client(timeout=10) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            normalized = response.json()["choices"][0]["message"]["content"].strip()
            logger.debug(f"Query expanded: '{query}' → '{normalized}'")
            return normalized
    except Exception as e:
        logger.warning(f"Query normalization failed (using original): {e}")
        return query


async def get_embeddings(
    cases: List[Dict[str, Any]], api_key: str
) -> Dict[str, List[float]]:
    """Return embeddings for all cases, using cache where possible.

    Cache key = case id. The entry stores {hash, embedding}.
    If the file's MD5 hash has changed, the embedding is regenerated.
    """
    cache = _load_cache()
    embeddings: Dict[str, List[float]] = {}
    updated = False

    for case in cases:
        cid = case["id"]
        cached = cache.get(cid, {})

        if cached.get("hash") == case["hash"] and cached.get("embedding"):
            embeddings[cid] = cached["embedding"]
            continue

        if not case["content"]:
            logger.debug(f"Skipping empty case: {cid}")
            continue

        logger.info(f"Generating embedding for case: {cid}")
        try:
            vec = await _embed_text(case["content"], api_key)
            embeddings[cid] = vec
            cache[cid] = {"hash": case["hash"], "embedding": vec}
            updated = True
        except Exception as e:
            logger.error(f"Failed to embed case {cid}: {e}")

    if updated:
        _save_cache(cache)

    return embeddings


# ─────────────────────────── Retrieval ───────────────────────────────────────

async def retrieve_cases(
    query: str,
    api_key: str,
    top_k: int = 5,
    threshold: float = 0.25,
    fallback_floor: float = 0.05,
) -> List[Dict[str, Any]]:
    """Find the most relevant case studies for a given query.

    Returns a list of case dicts (id, title, content, score), sorted by
    descending similarity. Only cases above `threshold` are returned.

    Fallback: if no case meets `threshold` (e.g. a generic "show me all
    cases" query or a cross-lingual query after normalization), the top_k
    results are returned anyway as long as their score exceeds `fallback_floor`.
    This ensures the user always gets something useful for browsing queries.

    Query normalization: non-ASCII queries (e.g. Russian) are translated to
    English keywords via gpt-4o-mini before embedding to improve cross-lingual
    retrieval quality.

    Empty placeholder cases are skipped.
    """
    cases = load_cases()
    if not cases:
        return []

    # Filter out empty placeholders
    populated = [c for c in cases if len(c["content"]) > 100]
    if not populated:
        return []

    embeddings = await get_embeddings(populated, api_key)
    if not embeddings:
        return []

    try:
        normalized_query = await _normalize_query(query, api_key)
        query_vec = await _embed_text(normalized_query, api_key)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []

    scored = []
    for case in populated:
        vec = embeddings.get(case["id"])
        if vec is None:
            continue
        score = _cosine_similarity(query_vec, vec)
        scored.append({**case, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Primary: cases above threshold
    above_threshold = [c for c in scored if c["score"] >= threshold]
    if above_threshold:
        return above_threshold[:top_k]

    # Fallback: return top_k even below threshold (e.g. "show me all cases")
    fallback = [c for c in scored if c["score"] >= fallback_floor]
    if fallback:
        logger.info(
            f"RAG: no cases above threshold {threshold} — using fallback "
            f"(top score: {scored[0]['score'] if scored else 'N/A'})"
        )
        return fallback[:top_k]

    return []


# ─────────────────────────── Metadata (for API listing) ──────────────────────

def get_cases_metadata() -> List[Dict[str, Any]]:
    """Return lightweight metadata for all cases (no embeddings, no full content).

    Used by GET /api/knowledge-base.
    """
    cases = load_cases()
    return [
        {
            "id": c["id"],
            "title": c["title"],
            "industry": c["industry"],        # list
            "tech_stack": c["tech_stack"],    # list
            "client_type": c["client_type"],  # "enterprise" | "scaleup" | "startup" | ""
            "region": c["region"],            # "DACH" | "UK" | "US" | etc.
            "is_populated": len(c["content"]) > 100,
        }
        for c in cases
    ]
