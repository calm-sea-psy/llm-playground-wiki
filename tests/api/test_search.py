from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from playground_corpus import Chunk
from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import SearchService

from playground_wiki.api.app import create_app
from playground_wiki.api.config import ApiConfig
from playground_wiki.api.schemas import MAX_K, MAX_QUERY_CHARS

CHUNK = Chunk(
    id="1-2",
    doc_id="1",
    title="서울",
    section="역사",
    text="서울 > 역사\n한양",
    seq=2,
    token_count=5,
)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, Filters | None]] = []

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        self.calls.append((query, k, filters))
        return [Hit(chunk=CHUNK, score=0.75)]


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def client(retriever: FakeRetriever) -> TestClient:
    app = create_app()  # with 없이 쓰므로 lifespan(모델 · DB) 은 돌지 않는다
    app.state.search_service = SearchService(retriever, collection="test", chunk_count=7)
    return TestClient(app)


def test_search_returns_hits(client: TestClient, retriever: FakeRetriever):
    response = client.post("/search", json={"query": "  조선의 수도는?  ", "k": 3})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["query"] == "조선의 수도는?"
    assert body["collection"] == "test"
    assert body["hits"] == [
        {
            "chunk_id": "1-2",
            "doc_id": "1",
            "title": "서울",
            "section": "역사",
            "text": "서울 > 역사\n한양",
            "score": 0.75,
        }
    ]
    assert retriever.calls == [("조선의 수도는?", 3, None)]


def test_search_passes_filters(client: TestClient, retriever: FakeRetriever):
    client.post("/search", json={"query": "수도", "doc_ids": ["1"], "categories": ["도시"]})

    assert retriever.calls[0][2] == Filters(
        doc_ids=frozenset({"1"}), categories=frozenset({"도시"})
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "가" * (MAX_QUERY_CHARS + 1)},
        {"query": "수도", "k": 0},
        {"query": "수도", "k": MAX_K + 1},
    ],
)
def test_search_rejects_invalid_request(
    client: TestClient, retriever: FakeRetriever, payload: dict[str, object]
):
    response = client.post("/search", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert retriever.calls == []


def test_health_reports_collection(client: TestClient):
    response = client.get("/health")

    assert response.json() == {"status": "ok", "collection": "test", "chunks": 7}


def test_config_reads_repo_configs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_CONFIG_DIR", str(Path(__file__).parents[2] / "configs"))
    monkeypatch.setenv("RAG_COLLECTION", "kowiki-subset-10000_kure-v1")

    config = ApiConfig.from_env()

    assert config.search.collection == "kowiki-subset-10000_kure-v1"
    assert config.search.model == "kure-v1"
    assert config.search.source == "kowiki"
    assert (config.search.candidates, config.search.rrf_k) == (50, 60)
    assert (config.search.rerank_candidates, config.search.top_k) == (20, 5)
    assert config.search.rerank_candidates >= MAX_K  # 요청 k 상한만큼 돌려줄 수 있어야 한다


def test_config_rejects_fewer_rerank_candidates_than_the_k_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = Path(__file__).parents[2] / "configs"
    for name in ("ingest.yaml", "retrieval.yaml"):
        (tmp_path / name).write_text((repo / name).read_text(encoding="utf-8"), encoding="utf-8")
    retrieval = tmp_path / "retrieval.yaml"
    text = retrieval.read_text(encoding="utf-8")
    retrieval.write_text(text.replace("top_k: 20", f"top_k: {MAX_K - 1}"), encoding="utf-8")
    monkeypatch.setenv("RAG_CONFIG_DIR", str(tmp_path))

    with pytest.raises(ValueError, match=r"fusion\.top_k"):
        ApiConfig.from_env()


def test_config_defaults_collection_to_template(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_CONFIG_DIR", str(Path(__file__).parents[2] / "configs"))
    monkeypatch.delenv("RAG_COLLECTION", raising=False)

    assert ApiConfig.from_env().search.collection == "kowiki_kure-v1"
