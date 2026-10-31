"""에이전트 노드의 LLM 출력 형식 (경계 모듈): `generate_json` 에 넘길 JSON 스키마 · 검증과 내부 값으로 바꾸기.

pydantic 모델은 이 파일 밖으로 넘기지 않는다 — 검증한 값은 바로 튜플 · `Judgment` 로 바꿔 돌려준다.
"""

from collections.abc import Sequence

from playground_llm import LLM, Message, ModelSpec
from playground_llm.structured import generate_json
from pydantic import BaseModel, ConfigDict

from playground_wiki.agent.state import Judgment


class DecomposeOutput(BaseModel):
    """질문 나누기 노드의 출력."""

    model_config = ConfigDict(frozen=True)

    sub_questions: list[str]


class JudgeOutput(BaseModel):
    """근거 판단 노드의 출력."""

    model_config = ConfigDict(frozen=True)

    sufficient: bool
    found: str
    missing: str
    queries: list[str]


def generate_sub_questions(
    llm: LLM, spec: ModelSpec, messages: Sequence[Message], *, limit: int
) -> tuple[str, ...]:
    """질문 나누기 노드를 불러 빈 글 · 겹친 것을 뺀 하위 질문을 앞에서부터 limit 개까지 돌려준다 (동기 호출).

    Raises:
      StructuredOutputError: 다시 요청해도 JSON 형식이 맞지 않을 때.
      GenerationError: LLM 호출이 실패했을 때.
      ModelNotFoundError: 서버에 그 모델이 없을 때.
    """
    output = generate_json(llm, spec, messages, DecomposeOutput).value
    return _cleaned(output.sub_questions)[:limit]


def generate_judgment(llm: LLM, spec: ModelSpec, messages: Sequence[Message]) -> Judgment:
    """근거 판단 노드를 불러 검증한 판단을 돌려준다 (동기 호출). 새 검색어는 빈 글 · 겹친 것을 뺀다.

    Raises:
      StructuredOutputError: 다시 요청해도 JSON 형식이 맞지 않을 때.
      GenerationError: LLM 호출이 실패했을 때.
      ModelNotFoundError: 서버에 그 모델이 없을 때.
    """
    output = generate_json(llm, spec, messages, JudgeOutput).value
    return Judgment(
        sufficient=output.sufficient,
        found=output.found.strip(),
        missing=output.missing.strip(),
        queries=_cleaned(output.queries),
        status="ok",
    )


def _cleaned(texts: Sequence[str]) -> tuple[str, ...]:
    kept: list[str] = []
    for text in texts:
        stripped = text.strip()
        if stripped and stripped not in kept:
            kept.append(stripped)
    return tuple(kept)
