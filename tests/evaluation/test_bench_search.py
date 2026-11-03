import asyncio
import hashlib
import json
from dataclasses import dataclass

from playground_corpus import Chunk
from playground_retrieval.core import Hit

from playground_wiki.evaluation.bench_search import (
    PROGRESS_EVERY,
    hit_row,
    search_all,
    search_settings,
)


@dataclass(frozen=True)
class FakeSettings:
    collection: str = "c"
    model: str = "m"
    candidates: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    reranker_model: str = "r"


def test_search_settings_keeps_keys_order_and_set_name_hash():
    # 세트 이름 해시 입력이다 — 키 · 값 · 차례가 바뀌면 세트 폴더 이름이 바뀐다
    settings = search_settings(FakeSettings(), top_k=5, search_k=100)

    assert list(settings) == [
        "collection",
        "model",
        "candidates",
        "rrf_k",
        "rerank_candidates",
        "reranker_model",
        "top_k",
        "search_k",
    ]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    assert digest == "01027217"


def test_search_settings_without_extra_ends_with_top_k():
    assert list(search_settings(FakeSettings(), top_k=5))[-1] == "top_k"


def test_search_all_keeps_order_and_reports_progress():
    calls: list[tuple[int, int]] = []

    async def search(query: str) -> str:
        return query.upper()

    queries = ["a"] * (PROGRESS_EVERY * 2 + 1)
    results = asyncio.run(
        search_all(search, queries, on_progress=lambda done, total: calls.append((done, total)))
    )

    assert results == ["A"] * len(queries)
    assert calls == [(PROGRESS_EVERY, len(queries)), (PROGRESS_EVERY * 2, len(queries))]


def test_hit_row_rounds_score():
    chunk = Chunk(
        id="d-0", doc_id="d", title="제목", section="절", text="본문", seq=0, token_count=1
    )

    assert hit_row(Hit(chunk=chunk, score=0.12345678)) == {
        "chunk_id": "d-0",
        "title": "제목",
        "section": "절",
        "text": "본문",
        "score": 0.123457,
    }
