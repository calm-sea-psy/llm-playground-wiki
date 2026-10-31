"""질의 변형 실험의 질의 만들기 — LLM 으로 검색용 고친 질의 1개(②)와 여러 질의 2~3개(③)를 만들어 문항마다 기록한다.

고친 질의는 후보를 모으는 데만 쓰고, 순위는 원문 질문으로 매긴다 (query_variants). 프롬프트는 답변 프롬프트
(generation/prompts)와 따로 이 패키지의 prompts/ 에 둔다 — 실험에서 이긴 변형만 에이전트로 옮긴다.
"""

import json
import logging
import re
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from playground_evalkit import squash
from playground_llm import (
    LLM,
    GenerationError,
    Message,
    ModelSpec,
    ParamValue,
    PromptLibrary,
    PromptTemplate,
)

logger = logging.getLogger(__name__)

QUERY_PROMPT_DIR = Path(__file__).parent / "prompts"
"""질의 변형 실험 프롬프트 (`{이름}/{버전}.md`). `PromptLibrary.load(QUERY_PROMPT_DIR)` 로 읽는다."""
REWRITE_PROMPT = "rewrite@v1"
MULTI_QUERY_PROMPT = "multi_query@v1"
# think off — 짧은 변환이라 생각이 필요 없고, 걸린 시간이 그대로 검색 앞에 붙는다. temperature 0 — 다시 돌려도 같은 질의.
# max_tokens 256 — 질의 세 개 JSON 이 한국어 100 토큰 안팎이라 넉넉하다. 모델은 쓰는 쪽이 답변 프로필에서 가져온다
REWRITE_PARAMS: Mapping[str, ParamValue] = MappingProxyType(
    {"think": "off", "temperature": 0.0, "max_tokens": 256, "num_ctx": 2048}
)
MULTI_QUERY_MAX = 3
"""여러 질의(원문 제외)의 최대 수 — 계획의 '변형 2~3개'. 더 내면 앞에서 자른다."""
MAX_CONSECUTIVE_ERRORS = 3
"""생성 실패가 이만큼 이어지면 Ollama 가 내려간 것 — 남은 문항을 실패로 채우지 않고 멈춘다 (answer_run 과 같은 값)."""
GENERATION_ERROR = "generation"
"""오류 기록의 머리 — 실행 환경의 실패라 다음 실행에서 다시 돈다. 형식 오류(format)는 모델이 낸 결과라 남긴다."""
FORMAT_ERROR = "format"
REWRITE_SCHEMA: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
)
MULTI_QUERY_SCHEMA: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": MULTI_QUERY_MAX,
            }
        },
        "required": ["queries"],
    }
)
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class Rewrite:
    """문항 하나의 고친 질의 · 여러 질의와 만든 시간."""

    id: str
    question: str
    """고쳐 쓴 원래 질문 — 검색 단계가 같은 질문의 기록인지 대조한다."""
    rewrite: str | None
    """검색용으로 고친 질의 (②). 만들지 못했으면 None — 검색 단계는 원문 하나로 대신한다."""
    queries: tuple[str, ...]
    """여러 질의 (③, 원문 제외 0~MULTI_QUERY_MAX 개). 빈 것 · 원문과 같은 것 · 서로 겹치는 것은 뺐다."""
    rewrite_ms: float
    """고쳐 쓰기 LLM 호출 시간."""
    multi_ms: float
    """여러 질의 LLM 호출 시간."""
    rewrite_error: str | None = None
    """`generation: …` 또는 `format: …`. 없으면 None."""
    multi_error: str | None = None

    @property
    def retryable(self) -> bool:
        """생성 실패(실행 환경)가 있어 다음 실행에서 다시 돌 기록인가."""
        return any(
            error is not None and error.startswith(GENERATION_ERROR)
            for error in (self.rewrite_error, self.multi_error)
        )

    def to_json(self) -> dict[str, Any]:
        """JSONL 한 줄."""
        return {
            "id": self.id,
            "question": self.question,
            "rewrite": self.rewrite,
            "queries": list(self.queries),
            "rewrite_ms": self.rewrite_ms,
            "multi_ms": self.multi_ms,
            "rewrite_error": self.rewrite_error,
            "multi_error": self.multi_error,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """JSONL 한 줄에서."""
        return cls(
            id=str(row["id"]),
            question=str(row["question"]),
            rewrite=row["rewrite"],
            queries=tuple(str(query) for query in row["queries"]),
            rewrite_ms=float(row["rewrite_ms"]),
            multi_ms=float(row["multi_ms"]),
            rewrite_error=row.get("rewrite_error"),
            multi_error=row.get("multi_error"),
        )


class QueryRewriter:
    """질문 하나에 LLM 을 두 번 불러(고쳐 쓰기 · 여러 질의) Rewrite 를 만든다.

    프롬프트는 system 에 지시만, user 에 질문만 넣는다. 형식이 틀린 답은 다시 묻지 않고 오류로 남긴다 —
    변형이 실제로 쓰일 때도 한 번 부르고 끝이라 그 실패율이 곧 비용이다.

    예)
        rewriter = QueryRewriter.from_library(llm, spec, PromptLibrary.load(QUERY_PROMPT_DIR))
        rewrite = rewriter.rewrite("k1", "경복궁을 다시 지은 사람은?")
    """

    def __init__(
        self,
        llm: LLM,
        spec: ModelSpec,
        *,
        rewrite_prompt: PromptTemplate,
        multi_prompt: PromptTemplate,
    ) -> None:
        self._llm = llm
        self.spec = spec
        self._rewrite_prompt = rewrite_prompt
        self._multi_prompt = multi_prompt

    @classmethod
    def from_library(cls, llm: LLM, spec: ModelSpec, library: PromptLibrary) -> Self:
        """프롬프트 모음에서 REWRITE_PROMPT · MULTI_QUERY_PROMPT 를 꺼내 만든다."""
        return cls(
            llm,
            spec,
            rewrite_prompt=library.get(REWRITE_PROMPT),
            multi_prompt=library.get(MULTI_QUERY_PROMPT),
        )

    @property
    def prompt_versions(self) -> tuple[str, str]:
        """(고쳐 쓰기, 여러 질의) 프롬프트의 `이름@버전:해시` — 실행 meta 에 남긴다."""
        return (
            f"{self._rewrite_prompt.ref}:{self._rewrite_prompt.digest}",
            f"{self._multi_prompt.ref}:{self._multi_prompt.digest}",
        )

    def rewrite(self, item_id: str, question: str) -> Rewrite:
        """질문의 고친 질의 · 여러 질의. 실패는 예외 대신 기록의 오류 칸으로 돌려준다."""
        rewrite, rewrite_ms, rewrite_error = None, 0.0, None
        try:
            text, rewrite_ms = self._ask(self._rewrite_prompt, REWRITE_SCHEMA, question)
            rewrite = parse_rewrite(text)
        except GenerationError as e:
            rewrite_error = f"{GENERATION_ERROR}: {e}"
        except ValueError as e:
            rewrite_error = f"{FORMAT_ERROR}: {e}"
        queries: tuple[str, ...] = ()
        multi_ms, multi_error = 0.0, None
        try:
            text, multi_ms = self._ask(self._multi_prompt, MULTI_QUERY_SCHEMA, question)
            queries = parse_queries(text, question)
        except GenerationError as e:
            multi_error = f"{GENERATION_ERROR}: {e}"
        except ValueError as e:
            multi_error = f"{FORMAT_ERROR}: {e}"
        return Rewrite(
            id=item_id,
            question=question,
            rewrite=rewrite,
            queries=queries,
            rewrite_ms=rewrite_ms,
            multi_ms=multi_ms,
            rewrite_error=rewrite_error,
            multi_error=multi_error,
        )

    def _ask(
        self, template: PromptTemplate, schema: Mapping[str, object], question: str
    ) -> tuple[str, float]:
        messages = [Message("system", template.render()), Message("user", question)]
        start = time.perf_counter()
        generation = self._llm.generate(self.spec, messages, json_schema=schema)
        return generation.text, round((time.perf_counter() - start) * 1000, 1)


def parse_rewrite(text: str) -> str:
    """LLM 답 `{"query": …}` 의 고친 질의 (앞뒤 공백을 뗀 것).

    Raises:
      ValueError: JSON 객체가 아니거나 query 가 빈 문자열 · 문자열이 아닐 때.
    """
    query = _json_object(text).get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query must be a non-empty string, got {query!r}")
    return query.strip()


def parse_queries(text: str, question: str) -> tuple[str, ...]:
    """LLM 답 `{"queries": […]}` 의 여러 질의 — 빈 것 · 원문과 같은 것 · 앞과 겹치는 것을 빼고 앞에서 MULTI_QUERY_MAX 개.

    같은지는 공백을 무시하고 본다 (`squash`).

    Raises:
      ValueError: JSON 객체가 아니거나, queries 가 문자열 목록이 아니거나, 빼고 나서 하나도 남지 않을 때.
    """
    queries = _json_object(text).get("queries")
    if not isinstance(queries, list) or not all(isinstance(query, str) for query in queries):
        raise ValueError(f"queries must be a list of strings, got {queries!r}")
    seen = {squash(question)}
    kept: list[str] = []
    for query in queries:
        key = squash(query)
        if key and key not in seen:
            seen.add(key)
            kept.append(query.strip())
    if not kept:
        raise ValueError("no query left after dropping blanks and copies of the question")
    return tuple(kept[:MULTI_QUERY_MAX])


def dropped_numbers(question: str, query: str) -> tuple[str, ...]:
    """원문 질문의 숫자(날짜 · 수량 · 순번) 중 고친 질의에 없는 것 — 조건이 빠졌는지 눈으로 볼 문항을 고른다."""
    have = set(_NUMBER.findall(query))
    return tuple(number for number in _NUMBER.findall(question) if number not in have)


def added_numbers(question: str, query: str) -> tuple[str, ...]:
    """고친 질의에 새로 생긴 숫자 — 원문에 없는 날짜 · 수치를 지어 넣었는지 눈으로 볼 문항을 고른다."""
    have = set(_NUMBER.findall(question))
    return tuple(number for number in _NUMBER.findall(query) if number not in have)


def read_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """이어 쓴 JSONL 의 id 별 마지막 줄. 파일이 없으면 빈 dict, 쓰다 만 마지막 줄(실행이 끊김)은 버린다.

    Raises:
      ValueError: 마지막이 아닌 줄이 JSON 이 아닐 때 — 끊김이 아니라 파일이 깨졌다.
    """
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index < len(lines) - 1:
                raise ValueError(f"{path}: line {index + 1} is not JSON") from None
            logger.warning("%s: dropped a half-written last line", path)
            continue
        rows[str(row["id"])] = row
    return rows


def read_rewrites(path: Path) -> dict[str, Rewrite]:
    """고친 질의 기록의 id 별 마지막 기록."""
    return {item_id: Rewrite.from_json(row) for item_id, row in read_rows_by_id(path).items()}


def pending_ids(ids: Sequence[str], done: Mapping[str, Rewrite] | Collection[str]) -> list[str]:
    """넘긴 id 중 아직 돌릴 것 (순서 그대로). 끝난 기록이라도 생성 실패(retryable)가 있으면 다시 돈다."""
    if isinstance(done, Mapping):
        finished = {item_id for item_id, rewrite in done.items() if not rewrite.retryable}
    else:
        finished = set(done)
    return [item_id for item_id in ids if item_id not in finished]


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from None
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")  # noqa: TRY004  # 모델 출력 형식 오류 — 호출자가 ValueError 하나로 받는다
    return value
