"""API 계약 산출물: 계약 파일 위치 · OpenAPI 문서 글 · 예시 JSON 검사. scripts/ 의 계약 도구 · 목업 · 시험이 같이 쓴다.

예시는 contracts/examples/*.json 이다. 파일 이름이 모델을 정한다 — `EXAMPLE_MODELS` 의 이름은 그 모델 하나,
`ask_*` 는 /ask 요청 하나와 그 응답 이벤트 열(`{"request": {"question", "mode"}, "events": [{"event", "data"}…]}`)이다.
예시는 모델로 읽고 다시 썼을 때 같은 JSON 이어야 한다 (모르는 키 · 바뀐 값이 없게).
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_args

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

from playground_wiki.api.answer import AnswerMode
from playground_wiki.api.schemas import (
    ASK_EVENTS,
    AskRequest,
    EventName,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)

CONTRACTS_DIR = Path(
    "contracts"
)  # 레포 루트 기준 (configs/ 와 같이). scripts/ 는 레포 루트를 앞에 붙인다
OPENAPI_FILE = CONTRACTS_DIR / "openapi.json"
EXAMPLES_DIR = CONTRACTS_DIR / "examples"
EXAMPLE_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "search_request": SearchRequest,
        "search_response": SearchResponse,
        "health": HealthResponse,
    }
)
"""예시 파일 이름(확장자 뺀) → 모델."""
ASK_EXAMPLE_PREFIX = "ask_"
"""/ask 요청 · 이벤트 열 예시의 파일 이름 앞말."""
TERMINAL_EVENTS: frozenset[EventName] = frozenset({"done", "error"})
"""스트림을 닫는 이벤트 — 마지막에 한 번만 온다."""


def openapi_text(app: FastAPI) -> str:
    """앱의 OpenAPI 문서를 contracts/openapi.json 에 쓰는 글로 (2칸 들여쓰기, 한글 그대로, 끝 줄바꿈).

    문서만 만들고 lifespan 은 돌리지 않는다 — `create_app()` 을 그대로 넘긴다.
    """
    return json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n"


def check_examples(directory: Path) -> dict[str, list[str]]:
    """폴더의 예시 파일마다 어긴 곳 목록을 돌려준다 (파일 이름순). 지킨 파일은 빈 목록."""
    return {path.name: check_example(path) for path in sorted(directory.glob("*.json"))}


def check_example(path: Path) -> list[str]:
    """예시 파일 하나가 계약을 지키는지 본다. 어긴 곳을 한 줄씩 돌려준다 (지키면 빈 목록)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"JSON 이 아니다: {e}"]
    if path.stem in EXAMPLE_MODELS:
        problem = round_trip_problem(EXAMPLE_MODELS[path.stem], data)
        return [] if problem is None else [problem]
    if path.stem.startswith(ASK_EXAMPLE_PREFIX):
        return _ask_problems(data)
    return [
        f"어느 모델의 예시인지 모른다 — {sorted(EXAMPLE_MODELS)} 또는 {ASK_EXAMPLE_PREFIX}* 로 이름 짓는다"
    ]


def round_trip_problem(model: type[BaseModel], data: object) -> str | None:
    """모델로 읽고 다시 쓴 JSON 이 data 와 다르면 그 까닭을 돌려준다. 같으면 None."""
    try:
        parsed = model.model_validate(data)
    except ValidationError as e:
        return f"{model.__name__} 검증 실패: {e.errors(include_url=False)}"
    if parsed.model_dump(mode="json", exclude_unset=True) != data:
        return f"{model.__name__} 로 읽고 다시 쓰면 달라진다 (모르는 키나 바뀐 값이 있다)"
    return None


def event_order_problems(events: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """/ask 이벤트 열 (이름, data) 이 순서 규칙을 지키는지 본다. 어긴 곳을 한 줄씩 돌려준다.

    규칙: done · error 는 마지막에 한 번만 온다. delta 앞에는 stage 가 하나 이상 있다.
    done 의 stage_path 는 그때까지 온 stage 이벤트의 stage 와 같다.
    """
    if not events:
        return ["이벤트가 없다"]
    problems: list[str] = []
    names = [name for name, _ in events]
    if names[-1] not in TERMINAL_EVENTS:
        problems.append(f"마지막 이벤트가 done · error 가 아니다: {names[-1]}")
    if early := [i for i, name in enumerate(names[:-1]) if name in TERMINAL_EVENTS]:
        problems.append(f"done · error 뒤에 이벤트가 더 온다 (위치 {early})")
    stages: list[str] = []
    for i, (name, data) in enumerate(events):
        if name == "stage":
            stages.append(str(data.get("stage")))
        elif name == "delta" and not stages:
            problems.append(f"stage 없이 delta 가 온다 (위치 {i})")
        elif name == "done" and data.get("stage_path") != stages:
            problems.append(
                f"done.stage_path {data.get('stage_path')} 가 stage 이벤트 {stages} 와 다르다"
            )
    return problems


def _ask_problems(data: object) -> list[str]:
    if not isinstance(data, dict) or set(data) != {"request", "events"}:
        return ['{"request": …, "events": […]} 꼴이 아니다']
    request, events = data["request"], data["events"]
    problems: list[str] = []
    if not isinstance(request, dict) or request.get("mode") not in get_args(AnswerMode):
        problems.append(f"request.mode 가 {get_args(AnswerMode)} 중 하나가 아니다")
    else:
        question = {key: value for key, value in request.items() if key != "mode"}
        if (problem := round_trip_problem(AskRequest, question)) is not None:
            problems.append(f"request: {problem}")
    if not isinstance(events, list):
        return [*problems, "events 가 목록이 아니다"]
    pairs: list[tuple[str, Mapping[str, Any]]] = []
    for i, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"event", "data"}:
            problems.append(f'events[{i}]: {{"event", "data"}} 꼴이 아니다')
            continue
        name, payload = event["event"], event["data"]
        if name not in ASK_EVENTS:
            problems.append(f"events[{i}]: 모르는 이벤트 {name!r}")
            continue
        if (problem := round_trip_problem(ASK_EVENTS[name], payload)) is not None:
            problems.append(f"events[{i}] {name}: {problem}")
        pairs.append((name, payload))
    return problems + event_order_problems(pairs)
