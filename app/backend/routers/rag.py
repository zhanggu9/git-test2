from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "investment_docs")
_RAG_LLM_TIMEOUT_SECONDS = max(30, int(os.getenv("RAG_LLM_TIMEOUT_SECONDS", "180")))
_ROOT_DIR = Path(__file__).resolve().parents[3]
_DOCS_DIR = _ROOT_DIR / "docs"
_EMBEDDING_DIM = 384
_CHUNK_SIZE = 1200
_CHUNK_OVERLAP = 200
_RRF_K = 60


def _qdrant_request(method: str, path: str, payload: dict | None = None) -> dict:
    url = _QDRANT_URL.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, method=method, headers=headers),
            timeout=10,
        ) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(502, f"Qdrant 오류({exc.code}): {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(503, f"Qdrant 연결 실패 ({_QDRANT_URL}): {exc.reason}") from exc


def _qdrant_available() -> bool:
    try:
        with urllib.request.urlopen(_QDRANT_URL.rstrip("/") + "/collections", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def _qdrant_collection_available() -> bool:
    try:
        with urllib.request.urlopen(
            f"{_QDRANT_URL.rstrip('/')}/collections/{_QDRANT_COLLECTION}",
            timeout=3,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _hash_embed(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """한국어·영문 토큰과 2/3-gram을 함께 사용하는 외부 모델 없는 384차원 임베딩."""
    vector = [0.0] * dim
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return vector

    tokens = re.findall(r"[\w가-힣]+", normalized)
    features: list[str] = list(tokens)
    compact = re.sub(r"\s+", "", normalized)
    for size in (2, 3):
        features.extend(
            compact[index:index + size]
            for index in range(max(0, len(compact) - size + 1))
        )

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big")
        index = value % dim
        vector[index] += 1.0 if value & 1 else -1.0

    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _embedding_method() -> str:
    return "hashing-384"


def _chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= _CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= _CHUNK_SIZE:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                piece = paragraph[start:start + _CHUNK_SIZE].strip()
                if piece:
                    chunks.append(piece)
                start += _CHUNK_SIZE
            current = ""

    if current:
        chunks.append(current)

    if _CHUNK_OVERLAP <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for index in range(1, len(chunks)):
        previous = chunks[index - 1]
        prefix = previous[-_CHUNK_OVERLAP:] if len(previous) > _CHUNK_OVERLAP else previous
        overlapped.append(f"{prefix}\n{chunks[index]}".strip())
    return overlapped


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = match.group(1)
            current_body = []
        else:
            current_body.append(line)

    sections.append((current_heading, current_body))
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def _document_chunks() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not _DOCS_DIR.exists():
        return records

    for path in sorted(_DOCS_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        chunk_index = 0
        for heading, body in _split_by_heading(content):
            if not body:
                continue
            for text in _chunk_text(body):
                records.append({
                    "source_doc": path.name,
                    "section": heading,
                    "chunk_index": chunk_index,
                    "text": text,
                })
                chunk_index += 1
    return records


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="검색 질문")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 최대 청크 수")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="최소 벡터 유사도 점수")


class RagAskRequest(RagSearchRequest):
    provider: str = Field(default="rag", pattern="^(rag|openai_compatible)$", description="답변 생성 모듈")


def _vector_search(query: str, top_k: int, score_threshold: float) -> list[dict[str, object]]:
    payload: dict[str, object] = {
        "vector": _hash_embed(query),
        "limit": top_k * 2,
        "with_payload": True,
        "with_vector": False,
    }
    if score_threshold > 0:
        payload["score_threshold"] = score_threshold

    result = _qdrant_request(
        "POST",
        f"/collections/{_QDRANT_COLLECTION}/points/search",
        payload,
    )

    rows = []
    for rank, hit in enumerate(result.get("result", []), start=1):
        payload_data = hit.get("payload", {})
        rows.append({
            "score": round(float(hit.get("score", 0)), 4),
            "source_doc": payload_data.get("source_doc", ""),
            "section": payload_data.get("section", ""),
            "chunk_index": payload_data.get("chunk_index", 0),
            "text": payload_data.get("text", ""),
            "_rank": rank,
        })
    return rows


def _keyword_search(query: str, top_k: int) -> list[dict[str, object]]:
    tokens = [t for t in re.findall(r"[\w가-힣]+", query.lower()) if len(t) >= 2]
    if not tokens:
        return []

    scored: list[dict[str, object]] = []
    for chunk in _document_chunks():
        normalized = str(chunk["text"]).lower()
        score = sum(normalized.count(token) for token in tokens)
        score += sum(2 for token in tokens if token in str(chunk["section"]).lower())
        if score > 0:
            scored.append({**chunk, "score": score})

    scored.sort(
        key=lambda item: (
            -int(item["score"]),
            str(item["source_doc"]),
            int(item["chunk_index"]),
        )
    )
    results = scored[:top_k * 2]
    for rank, item in enumerate(results, start=1):
        item["_rank"] = rank
    return results


def _rrf_merge(
    vector_results: list[dict[str, object]],
    keyword_results: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, object]] = {}

    def item_key(item: dict[str, object]) -> str:
        return f"{item.get('source_doc')}::{item.get('chunk_index')}"

    for item in vector_results + keyword_results:
        key = item_key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + int(item["_rank"]))
        merged.setdefault(key, item)

    ordered = sorted(scores, key=scores.get, reverse=True)
    result: list[dict[str, object]] = []
    for key in ordered[:top_k]:
        item = dict(merged[key])
        item["score"] = round(scores[key], 6)
        item.pop("_rank", None)
        result.append(item)
    return result


def _search(query: str, top_k: int, score_threshold: float) -> list[dict[str, object]]:
    return _rrf_merge(
        _vector_search(query, top_k, score_threshold),
        _keyword_search(query, top_k),
        top_k,
    )


def _require_qdrant() -> None:
    if not _qdrant_available():
        raise HTTPException(503, f"Qdrant 서버에 연결할 수 없습니다 ({_QDRANT_URL}). 벡터 DB를 먼저 실행하세요.")
    if not _qdrant_collection_available():
        raise HTTPException(503, "RAG 컬렉션이 아직 없습니다. Docker Compose의 docs-index를 먼저 실행하세요.")


def _cite(chunk: dict[str, object]) -> str:
    doc = str(chunk.get("source_doc", ""))
    section = str(chunk.get("section", "") or "").strip()
    return f"{doc} § {section}" if section else doc


def _rag_only_answer(chunks: list[dict[str, object]]) -> str:
    if not chunks:
        return "관련 문서를 찾지 못했습니다. 다른 표현으로 질문해 보세요."

    excerpts = []
    for chunk in chunks[:3]:
        text = " ".join(str(chunk.get("text", "")).split())
        if text:
            excerpts.append(f"• [{_cite(chunk)}] {text[:500]}{'…' if len(text) > 500 else ''}")

    return "문서에서 찾은 관련 내용입니다. 오른쪽 검색 근거에서 원문을 확인할 수 있습니다.\n\n" + "\n\n".join(excerpts)


def _openai_compatible_answer(query: str, chunks: list[dict[str, object]]) -> str:
    api_key = os.getenv("RAG_LLM_API_KEY")
    model = os.getenv("RAG_LLM_MODEL")
    base_url = os.getenv("RAG_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    if not api_key or not model:
        raise HTTPException(503, "외부 AI 요약을 사용하려면 RAG_LLM_API_KEY와 RAG_LLM_MODEL을 설정하세요.")

    context = "\n\n".join(
        f"[출처 {index + 1}: {_cite(chunk)}]\n{chunk.get('text', '')}"
        for index, chunk in enumerate(chunks)
    )[:14000]

    prompt = (
        "당신은 금융 교육용 RAG 어시스턴트입니다. "
        "아래 검색 원문만 근거로 사용자의 질문에 한국어로 답하세요. "
        "특정 종목의 매수·매도를 권유하지 말고 금융 개념, 상품 구조, 리스크와 학습 관점 중심으로 설명하세요. "
        "원문에 없는 사실·숫자를 추가하지 말고, 근거가 부족하면 "
        "'확인 필요 — 추가 자료나 최신 공시를 확인하세요'라고 답하세요. "
        "출처 번호를 [출처 1]처럼 표시하세요. "
        "답변 마지막에 '투자 판단과 책임은 투자자 본인에게 있습니다'를 덧붙이세요.\n\n"
        f"사용자 질문: {query}\n\n검색 원문:\n{context}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "제공된 검색 원문만 근거로 설명하는 금융 교육 RAG 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )

    try:
        with urllib.request.urlopen(request, timeout=_RAG_LLM_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = str(result.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not answer:
            raise ValueError("빈 응답")
        return answer
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(502, f"외부 AI 응답 오류({exc.code}): {detail[:200]}") from exc
    except (urllib.error.URLError, ValueError, KeyError, IndexError) as exc:
        raise HTTPException(502, f"외부 AI 응답을 받지 못했습니다: {exc}") from exc


@router.post("/api/rag/search")
def rag_search(req: RagSearchRequest) -> dict[str, object]:
    """Qdrant 벡터 검색과 로컬 키워드 검색을 RRF로 결합합니다."""
    _require_qdrant()
    chunks = _search(req.query, req.top_k, req.score_threshold)
    return {
        "query": req.query,
        "embed_method": _embedding_method(),
        "search_method": "hybrid_rrf",
        "count": len(chunks),
        "results": chunks,
    }


@router.post("/api/rag/ask")
def rag_ask(req: RagAskRequest) -> dict[str, object]:
    _require_qdrant()
    chunks = _search(req.query, req.top_k, req.score_threshold)
    answer = _rag_only_answer(chunks) if req.provider == "rag" else _openai_compatible_answer(req.query, chunks)
    return {
        "query": req.query,
        "answer": answer,
        "provider": req.provider,
        "embed_method": _embedding_method(),
        "search_method": "hybrid_rrf",
        "sources": chunks,
        "source_count": len(chunks),
    }


@router.get("/api/rag/status")
def rag_status() -> dict[str, object]:
    available = _qdrant_available()
    collection_available = available and _qdrant_collection_available()
    info: dict[str, object] = {}

    if collection_available:
        try:
            result = _qdrant_request("GET", f"/collections/{_QDRANT_COLLECTION}").get("result", {})
            info = {
                "points_count": result.get("points_count", 0),
                "vector_size": result.get("config", {}).get("params", {}).get("vectors", {}).get("size"),
                "status": result.get("status", "unknown"),
            }
        except Exception:
            info = {"error": "컬렉션이 없거나 조회 실패"}

    return {
        "qdrant": {
            "available": available,
            "collection_available": collection_available,
            "url": _QDRANT_URL,
            "collection": _QDRANT_COLLECTION,
            **info,
        },
        "external_ai": {
            "openai_compatible_available": bool(os.getenv("RAG_LLM_API_KEY") and os.getenv("RAG_LLM_MODEL"))
        },
        "embedding": {"provider": "hash", "model": "hashing-384"},
        "embed_method": _embedding_method(),
    }
}