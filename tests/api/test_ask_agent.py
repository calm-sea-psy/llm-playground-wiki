import json
from datetime import date

from fastapi import status
from fastapi.testclient import TestClient
from playground_corpus import Chunk
from playground_llm import FakeLLM, ModelSpec, PromptLibrary
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.agent import AgentRunner
from playground_wiki.api.answer import ANSWER_MODES, AnswerService
from playground_wiki.api.app import create_app
from playground_wiki.api.schemas import MAX_QUERY_CHARS
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, Profile, Thresholds

LIBRARY = PromptLibrary.load(PROMPT_DIR)
SPEC = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": 64})
GROUNDED = Profile(SPEC, system=("base@v4", "grounded@v5"), user="question@v1", timeout_sec=5)
PARTIAL = Profile(SPEC, system=("base@v4", "partial@v4"), user="question@v1", timeout_sec=5)
POLICY = AnswerPolicy(Thresholds(high=0.9, low=0.5), {"grounded": GROUNDED, "partial": PARTIAL})


class FakeSearch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str) -> SearchResult:
        self.calls.append(query)
        chunk = Chunk(
            id="1-0",
            doc_id="1",
            title="경복궁",
            section="",
            text="흥선대원군",
            seq=0,
            token_count=1,
        )
        return SearchResult(hits=(Hit(chunk=chunk, score=0.95),), took_ms=1.0)


def _reply(spec: ModelSpec, messages: list) -> str:
    del spec  # 노드는 메시지로 알아본다
    user = messages[-1].content
    if "## 할 일: 질문 나누기" in user:
        return json.dumps({"sub_questions": ["경복궁 중건"]}, ensure_ascii=False)
    if "## 할 일: 모은 문서로" in user:
        return json.dumps({"sufficient": True, "found": "", "missing": "", "queries": []})
    return "찾음\n흥선대원군이 중건했다 [1]."


def _client() -> tuple[TestClient, FakeSearch]:
    search = FakeSearch()
    llm = FakeLLM([_reply])
    agent = AgentRunner.from_spec(
        search=search, llm=llm, spec=SPEC, low=0.5, max_query_chars=MAX_QUERY_CHARS
    )
    app = create_app()
    app.state.answer_service = AnswerService(
        search=search,
        policy=POLICY,
        library=LIBRARY,
        llm=llm,
        gate=None,
        dump_date="2026년 10월 1일",
        today=lambda: date(2026, 10, 24),
        agent=agent,
    )
    return TestClient(app), search  # with 없이 — lifespan(모델 · 저장소)을 돌리지 않는다


def _events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def _stages(body: str) -> list[str]:
    return [data["stage"] for name, data in _events(body) if name == "stage"]


def _text(body: str) -> str:
    return "".join(data["text"] for name, data in _events(body) if name == "delta")


def test_modes_are_single_and_agent():
    assert ANSWER_MODES == ("single", "agent")


def test_ask_route_runs_the_agent_with_mode_agent():
    client, search = _client()

    response = client.post("/ask?mode=agent", json={"question": "경복궁을 중건한 사람은?"})

    assert response.status_code == status.HTTP_200_OK
    assert _stages(response.text) == ["agent_decompose", "agent_search", "agent_judge", "grounded"]
    assert search.calls == ["경복궁 중건"]
    assert _text(response.text) == "흥선대원군이 중건했다 [1]."


def test_ask_route_without_mode_is_the_single_path():
    client, search = _client()

    response = client.post("/ask", json={"question": "경복궁을 중건한 사람은?"})

    assert _stages(response.text) == ["grounded"]
    assert search.calls == ["경복궁을 중건한 사람은?"]


def test_ask_route_rejects_an_unknown_mode():
    client, search = _client()

    response = client.post("/ask?mode=multi", json={"question": "경복궁을 중건한 사람은?"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert search.calls == []
