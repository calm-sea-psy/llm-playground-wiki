import asyncio
import json
from pathlib import Path

import pytest
from helpers import NoTokens

from playground_wiki.api.app import create_app
from playground_wiki.api.config import AnswerSetup, ApiConfig
from playground_wiki.api.contract import (
    EXAMPLE_MODELS,
    check_example,
    event_order_problems,
    openapi_text,
    round_trip_problem,
)
from playground_wiki.api.mock import MOCK_SEARCH_EXAMPLE, MockServices
from playground_wiki.api.openapi import SSE_MEDIA_TYPE, with_event_schemas
from playground_wiki.api.schemas import ASK_EVENTS, StageEvent
from playground_wiki.api.version import CONTRACT_VERSION

ROOT = Path(__file__).parents[2]
CONTRACTS = ROOT / "contracts"
EXAMPLES = sorted((CONTRACTS / "examples").glob("*.json"))
ASK_EXAMPLES = [path for path in EXAMPLES if path.stem.startswith("ask_")]


@pytest.fixture
def mock(monkeypatch: pytest.MonkeyPatch) -> MockServices:
    monkeypatch.setenv("RAG_CONFIG_DIR", str(ROOT / "configs"))
    setup = AnswerSetup.from_config(ApiConfig.from_env(), tokenizer=NoTokens())
    return MockServices.from_config(setup, top_k=5, example=ROOT / MOCK_SEARCH_EXAMPLE, delay_sec=0)


def _shape(events: list[tuple[str, dict]]) -> list[str]:
    """이벤트 열의 모양 — stage 는 단계 이름까지, 이어진 delta 는 하나로 (조각 수는 예시에서 묶었다)."""
    shape: list[str] = []
    for name, data in events:
        label = f"stage:{data['stage']}" if name == "stage" else name
        if not (label == "delta" and shape and shape[-1] == "delta"):
            shape.append(label)
    return shape


@pytest.mark.parametrize("path", EXAMPLES, ids=[path.name for path in EXAMPLES])
def test_examples_follow_the_contract(path: Path):
    assert check_example(path) == []


@pytest.mark.parametrize("path", ASK_EXAMPLES, ids=[path.name for path in ASK_EXAMPLES])
def test_ask_examples_are_what_the_mock_server_sends(mock: MockServices, path: Path):
    # /ask 예시는 실제 답변 흐름(목업 검색 · FakeLLM)이 그 요청에 보내는 이벤트 열이다
    example = json.loads(path.read_text(encoding="utf-8"))
    request = example["request"]

    async def collect() -> list[tuple[str, dict]]:
        stream = mock.answer.answer(request["question"], mode=request["mode"])
        return [(event.name, dict(event.data)) async for event in stream]

    events = asyncio.run(collect())

    assert _shape(events) == _shape([(e["event"], e["data"]) for e in example["events"]])
    text = "".join(data["text"] for name, data in events if name == "delta")
    expected = "".join(e["data"]["text"] for e in example["events"] if e["event"] == "delta")
    assert text == expected


def test_examples_cover_every_model_and_ask_case():
    names = {path.stem for path in EXAMPLES}

    assert set(EXAMPLE_MODELS) <= names
    assert {
        "ask_gate",
        "ask_refuse",
        "ask_grounded",
        "ask_grounded_to_partial",
        "ask_error",
    } <= names
    assert any(name.startswith("ask_agent_") for name in names)


def test_committed_openapi_matches_the_app():
    # 스키마 · 라우터를 고치면 scripts/build_contract_manifest.py 로 다시 만든다
    committed = (CONTRACTS / "openapi.json").read_text(encoding="utf-8").replace("\r\n", "\n")

    assert committed == openapi_text()


def test_openapi_describes_ask_events_and_their_models():
    openapi = json.loads(openapi_text())
    content = openapi["paths"]["/ask"]["post"]["responses"]["200"]["content"]
    items = content[SSE_MEDIA_TYPE]["itemSchema"]["oneOf"]

    assert list(content) == [SSE_MEDIA_TYPE]  # application/json 이 덧붙지 않는다
    assert [item["properties"]["event"]["const"] for item in items] == list(ASK_EVENTS)
    refs = [item["properties"]["data"]["contentSchema"]["$ref"] for item in items]
    assert refs == [f"#/components/schemas/{model.__name__}" for model in ASK_EVENTS.values()]
    assert {model.__name__ for model in ASK_EVENTS.values()} <= set(
        openapi["components"]["schemas"]
    )
    assert openapi["info"]["version"] == CONTRACT_VERSION


def test_openapi_does_not_need_the_lifespan():
    # lifespan(저장소 · 모델 · Ollama)을 돌리지 않고 문서를 만든다 — app.state 가 비어 있다
    app = create_app()

    assert "/ask" in app.openapi()["paths"]
    assert not hasattr(app.state, "answer_service")


def test_event_schemas_refuse_a_colliding_component():
    openapi = {"components": {"schemas": {"StageEvent": {"type": "string"}}}}

    with pytest.raises(ValueError, match="StageEvent"):
        with_event_schemas(openapi, ASK_EVENTS)


def test_round_trip_catches_an_unknown_key_and_a_wrong_value():
    assert round_trip_problem(StageEvent, {"stage": "gate", "message": "m"}) is None
    assert round_trip_problem(StageEvent, {"stage": "gate", "message": "m", "extra": 1}) is not None
    assert round_trip_problem(StageEvent, {"stage": "nowhere", "message": "m"}) is not None


def test_event_order_rules():
    stage = ("stage", {"stage": "grounded", "message": "m"})
    delta = ("delta", {"text": "답"})
    done = ("done", {"stage_path": ["grounded"]})
    error = ("error", {"code": "internal", "message": "m"})

    assert event_order_problems([stage, delta, done]) == []
    assert event_order_problems([error]) == []
    assert event_order_problems([stage, delta])  # 끝 이벤트가 없다
    assert event_order_problems([delta, stage, done])  # stage 없이 delta
    assert event_order_problems([stage, done, delta])  # done 뒤에 이벤트
    assert event_order_problems([stage, stage, delta, done])  # stage_path 와 다르다
    assert event_order_problems([])


def test_check_example_rejects_an_unknown_file_name(tmp_path: Path):
    path = tmp_path / "query_response.json"
    path.write_text("{}", encoding="utf-8")

    assert check_example(path)


def test_check_example_reports_a_broken_ask_sequence(tmp_path: Path):
    path = tmp_path / "ask_broken.json"
    doc = {
        "request": {"question": "질문", "mode": "single"},
        "events": [{"event": "delta", "data": {"text": "답"}}],
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    problems = check_example(path)

    assert any("delta" in problem for problem in problems)
    assert any("done" in problem for problem in problems)
