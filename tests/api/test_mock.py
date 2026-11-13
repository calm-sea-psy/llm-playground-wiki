from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from playground_llm import FakeLLM, Message, ModelSpec, StreamDelta
from playground_retrieval.core import Filters

from playground_wiki.api.app import create_app
from playground_wiki.api.config import ApiConfig
from playground_wiki.api.mock import (
    MOCK_SEARCH_EXAMPLE,
    PARTIAL_TAG,
    REFUSE_TAG,
    SEARCH_ERROR_TAG,
    MockRetriever,
    PacedLLM,
)
from playground_wiki.generation import Thresholds

ROOT = Path(__file__).parents[2]
SEARCH_EXAMPLE = ROOT / MOCK_SEARCH_EXAMPLE
THRESHOLDS = Thresholds(high=0.8, low=0.1)


def test_mock_retriever_returns_the_example_hits_by_default():
    retriever = MockRetriever.from_example(SEARCH_EXAMPLE, THRESHOLDS)

    hits = retriever.search("경복궁을 중건한 사람은?", k=2)

    assert [hit.chunk.id for hit in hits] == ["1001-3", "2002-1"]
    assert hits[0].score == pytest.approx(0.97)
    assert hits[0].chunk.seq == 3


def test_mock_retriever_tags_pick_the_score_band():
    retriever = MockRetriever.from_example(SEARCH_EXAMPLE, THRESHOLDS)

    partial = retriever.search(f"질문 {PARTIAL_TAG}", k=5)

    assert THRESHOLDS.low <= partial[0].score < THRESHOLDS.high
    assert all(hit.score >= THRESHOLDS.low for hit in partial)  # 근거에서 빠지지 않는다
    assert retriever.search(f"질문 {REFUSE_TAG}", k=5) == []
    with pytest.raises(RuntimeError):
        retriever.search(f"질문 {SEARCH_ERROR_TAG}", k=5)


def test_mock_retriever_keeps_only_the_filtered_documents():
    retriever = MockRetriever.from_example(SEARCH_EXAMPLE, THRESHOLDS)

    hits = retriever.search("질문", k=5, filters=Filters(doc_ids=frozenset({"1001"})))

    assert {hit.chunk.doc_id for hit in hits} == {"1001"}


def test_mock_retriever_needs_a_grounded_top_score():
    with pytest.raises(ValueError, match="high"):
        MockRetriever.from_example(SEARCH_EXAMPLE, Thresholds(high=0.99, low=0.1))


def test_paced_llm_streams_the_same_pieces():
    llm = PacedLLM(FakeLLM(["경복궁은 흥선대원군이"]), delay_sec=0)
    spec = ModelSpec("ollama", "fake", {"think": "off"})
    messages = [Message("user", "질문")]

    pieces = [item.text for item in llm.stream(spec, messages) if isinstance(item, StreamDelta)]

    assert "".join(pieces) == "경복궁은 흥선대원군이"
    assert llm.generate(spec, messages).text == "경복궁은 흥선대원군이"


def test_impl_must_be_real_or_mock(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_CONFIG_DIR", str(ROOT / "configs"))
    assert ApiConfig.from_env().impl == "real"

    monkeypatch.setenv("RAG_IMPL", "fake")
    with pytest.raises(ValueError, match="RAG_IMPL"):
        ApiConfig.from_env()


def test_server_starts_in_mock_mode_without_stores_or_models(monkeypatch: pytest.MonkeyPatch):
    # lifespan 을 돌린다 — PG · Qdrant · Ollama · 검색 모델에 접속하지 않고 Kiwi(관문)만 올린다
    monkeypatch.chdir(ROOT)  # 설정 · 예시를 레포 루트 기준 경로로 읽는다
    monkeypatch.setenv("RAG_IMPL", "mock")
    monkeypatch.delenv("RAG_CONFIG_DIR", raising=False)

    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        search = client.post("/search", json={"query": "경복궁", "k": 2}).json()
        ask = client.post("/ask", json={"question": "오늘 날씨 어때"})

    assert health == {"status": "ok", "collection": "mock", "chunks": 5}
    assert [hit["chunk_id"] for hit in search["hits"]] == ["1001-3", "2002-1"]
    assert ask.text.startswith('event: stage\ndata: {"stage": "gate"')  # 실제 관문 (실시간 규칙)
    assert "event: done" in ask.text
