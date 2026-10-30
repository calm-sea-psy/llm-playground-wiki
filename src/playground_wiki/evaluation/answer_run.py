"""F9 답변 평가 실행: 평가 문항을 실제 답변 경로(`/ask` 와 같은 AnswerService)로 돌려 문항별 기록을 JSONL 에 이어 쓴다.

evaluation 은 api 를 import 하지 않는다 — 답변 서비스는 `Answerer` 계약으로 받고, 서비스 조립은 scripts/ 가 한다.
SSE 이벤트(stage · sources · delta · done · error, ADR-027)를 문항마다 기록 한 줄로 모은다. sources 이벤트에는 청크
본문이 없어서 검색기를 감싸 답변 경로가 받은 검색 결과를 같이 남긴다 — 정답 청크를 출처 번호로 짚었는지 보고,
나중에 지어냄을 판정할 때 문맥을 다시 검색하지 않으려고.
"""

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from playground_evalkit import KorQuADItem
from playground_retrieval.search import SearchResult

logger = logging.getLogger(__name__)

ItemSource = Literal["korquad", "offwiki", "f4", "harmful"]
"""korquad · offwiki 는 F9 표본, f4 는 F4 벤치 세트 문항을 저장된 문맥으로 다시 잰 것 (`f4_prose`),
harmful 은 유해 정보 세트(유해 질문과 짝지은 대조 질문 — scripts/judge_partial.py)."""
RETRY_ERRORS = frozenset({"generation"})
"""다음 실행에서 다시 돌릴 error 코드. 생성 실패(Ollama 접속 · 모델 없음)는 답변 경로가 아니라 실행 환경의 실패다.
시간 초과(timeout)는 답변 경로가 낸 결과라 기록으로 남긴다."""
MAX_CONSECUTIVE_ERRORS = (
    3  # 생성 실패가 이만큼 이어지면 Ollama 가 내려간 것 — 남은 문항을 실패로 채우지 않고 멈춘다
)
TAG_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
"""프롬프트 · 모델을 바꿔 다시 잰 기록의 꼬리표 (예 grounded-v3) — 파일 이름 · 리포트 이름에 그대로 들어간다."""


class Event(Protocol):
    """SSE 이벤트 하나 (api.answer.AnswerEvent)."""

    @property
    def name(self) -> str:
        """이벤트 이름: stage · sources · delta · done · error."""
        ...

    @property
    def data(self) -> Mapping[str, object]:
        """이벤트 내용 (JSON 으로 나가는 값)."""
        ...


class Searcher(Protocol):
    """답변 경로가 부르는 검색 (SearchService)."""

    async def search(self, query: str) -> SearchResult:
        """리랭커 점수 내림차순 상위 청크."""
        ...


class Answerer(Protocol):
    """질문 하나에 SSE 이벤트를 흘린다 (api.answer.AnswerService)."""

    def answer(self, question: str) -> AsyncIterator[Event]:
        """질문 하나의 이벤트: stage → (sources) → delta… → done, 또는 아무것도 못 보낸 실패면 error."""
        ...


@dataclass(frozen=True)
class EvalItem:
    """평가 문항 하나 (f9-sample JSONL 한 줄)."""

    id: str
    source: ItemSource
    question: str
    gold_title: str | None = None
    """KorQuAD 정답 문서 제목 (코퍼스 제목으로 보정한 것). F4 문항이면 세트의 정답 문서 제목. 위키 밖 질문이면 None."""
    gold_answer: str | None = None
    """KorQuAD 첫 번째 정답 문자열. F4 문항이면 세트의 첫 번째 정답 표기. 위키 밖 질문이면 None."""
    expected: str | None = None
    """위키 밖 질문의 기대 단계 (partial · refuse — 위키 내용으로 붙인 라벨). KorQuAD 면 None."""
    category: str | None = None
    """위키 밖 질문의 종류 (personal · realtime …), F4 문항이면 세트의 kind (no_doc · counterfactual …),
    유해 정보 세트면 harmful · contrast. KorQuAD 면 None."""
    trap: bool = False
    """위키 밖 질문 중 이름이 같은 문서가 있어 검색에 걸리기 쉬운 것."""

    def __post_init__(self) -> None:
        if self.source in {"korquad", "f4"} and (
            self.gold_title is None or self.gold_answer is None
        ):
            raise ValueError(f"{self.source} item needs gold_title and gold_answer, got {self.id=}")

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """f9-sample JSONL 한 줄(또는 기록의 item)에서. 모르는 키(문단 · related_title)는 무시한다."""
        return cls(
            id=str(row["id"]),
            source=cast("ItemSource", row["source"]),
            question=str(row["question"]),
            gold_title=row.get("gold_title"),
            gold_answer=row.get("gold_answer"),
            expected=row.get("expected"),
            category=row.get("category"),
            trap=bool(row.get("trap", False)),
        )

    def korquad(self) -> KorQuADItem | None:
        """정답 판정(`contains_answer`)에 넘길 KorQuAD 문항. 위키 밖 질문이면 None."""
        if self.gold_title is None or self.gold_answer is None:
            return None
        return KorQuADItem(
            id=self.id,
            title=self.gold_title,
            question=self.question,
            answer=self.gold_answer,
            source_title=self.gold_title,
        )


@dataclass(frozen=True)
class ContextChunk:
    """답변 경로가 받은 검색 결과 한 건. 리랭커 순위 n 이 문맥 안이면 출처 번호도 n 이다."""

    chunk_id: str
    title: str
    section: str
    text: str
    """청크 본문 ("제목 > 섹션" 머리줄 포함). 기록은 로컬에만 둔다."""
    score: float
    """리랭커 점수 (0~1)."""


@dataclass(frozen=True)
class AnswerRecord:
    """문항 하나를 답변 경로로 돌린 기록. f9-answers JSONL 한 줄."""

    item: EvalItem
    stage_path: tuple[str, ...]
    """거친 단계 (stage 이벤트 순서): gate · refuse · grounded · partial."""
    answer: str
    """delta 를 이어 붙인 답 원문 (관문 · 거절 고정 문구 포함). 표지는 서버가 뗀 뒤다."""
    hits: tuple[ContextChunk, ...]
    """답변 경로가 받은 검색 결과 (리랭커 점수 내림차순). 관문이 받았으면 비어 있다."""
    context_count: int
    """그중 문맥으로 넘긴 앞쪽 개수 (sources 이벤트 길이). 출처 번호 n 은 hits[n - 1]."""
    marker: str | None
    """grounded 첫 줄 표지: found · not_found · missing. grounded 를 거치지 않았거나 실패로 끝났으면 None."""
    cited: tuple[int, ...]
    """답에 남은 출처 번호 (처음 나온 순서, 중복 없이)."""
    dropped: tuple[int, ...]
    """범위를 벗어나 뗀 출처 번호 (나온 순서 그대로)."""
    truncated: bool
    """done 의 truncated — 마지막 단계의 생성이 max_tokens 에 닿았다. 고정 거절 문구로 끝났으면 False.
    단계마다의 잘림은 done_reason 에 있다."""
    timed_out: bool
    """답을 낸 단계가 시간 제한에 걸려 보낸 데까지로 끝났다."""
    error: str | None
    """error 이벤트의 code (generation · timeout). done 으로 끝났으면 None."""
    top_score: float | None
    """리랭커 1위 점수. 관문이 받았거나 검색 결과가 없으면 None."""
    gate_rule: str | None
    """관문이 받았으면 걸린 규칙 이름."""
    spec_id: str | None
    """마지막으로 생성한 단계의 모델 설정. LLM 을 부르지 않았거나 실패로 끝났으면 None."""
    prompt_version: str | None
    """마지막으로 생성한 단계의 프롬프트 버전. spec_id 와 같은 때 None."""
    search_ms: float | None
    """done 진단의 검색 시간 (ms). 관문이 받았거나 실패로 끝났으면 None."""
    first_delta_ms: float | None
    """done 진단의 첫 답 조각까지 시간 (질문을 받은 때부터, ms). LLM 이 답을 보내지 않았으면 None."""
    total_ms: float | None
    """done 진단의 전체 시간 (ms). 실패로 끝났으면 None."""
    wall_ms: float
    """평가 쪽에서 잰 질문 하나의 전체 시간 (ms). 실패로 끝나도 있다."""
    done_reason: Mapping[str, str | None] = field(default_factory=dict)
    """done 진단의 단계 → 생성이 멈춘 이유 (stop · length, 끊었거나 서버가 주지 않았으면 None). grounded 가 잘려
    partial 로 넘어간 것도 남는다. 실패로 끝난 기록과 이 필드가 생기기 전의 기록은 비어 있다."""

    def to_json(self) -> dict[str, Any]:
        """JSONL 한 줄로 쓸 값."""
        return asdict(self)

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """f9-answers JSONL 한 줄에서."""
        return cls(
            **{
                **row,
                "item": EvalItem.from_json(row["item"]),
                "stage_path": tuple(row["stage_path"]),
                "hits": tuple(ContextChunk(**hit) for hit in row["hits"]),
                "cited": tuple(row["cited"]),
                "dropped": tuple(row["dropped"]),
            }
        )


@dataclass(frozen=True)
class RunSummary:
    """한 번 실행한 결과."""

    total: int
    """입력 문항 수."""
    skipped: int
    """이미 기록이 있어 건너뛴 문항 수."""
    recorded: int
    """이번에 기록한 문항 수."""
    retried: int
    """다시 돌리려고 파일에서 지운 생성 실패 기록 수."""


class AnswerRun:
    """평가 문항을 답변 경로로 돌려 문항마다 기록 한 줄을 JSONL 에 이어 쓴다. 기록이 있는 문항은 건너뛴다.

    답변 서비스는 검색기를 받아 만드는 함수로 받는다 — 검색 결과(청크 본문)를 기록에 실으려고 검색기를 감싸 넘긴다.
    before_answer 는 문항마다 답변 경로를 부르기 직전에 부른다 — 문항마다 저장된 문맥을 돌려주는 검색기(F4 문항)처럼
    문항에 맞춰 바꿔 끼울 것이 있을 때.

    예)
        run = AnswerRun(search_service, lambda search: AnswerService(search=search, ...))
        summary = asyncio.run(run.run(items, Path("data/eval/f9-answers-v1.jsonl")))
    """

    def __init__(
        self,
        search: Searcher,
        make_answerer: Callable[[Searcher], Answerer],
        *,
        before_answer: Callable[[EvalItem], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._search = _RecordingSearch(search)
        self._answerer = make_answerer(self._search)
        self._before_answer = before_answer
        self._clock = clock

    async def run(
        self,
        items: Sequence[EvalItem],
        path: Path,
        *,
        limit: int | None = None,
        on_record: Callable[[int, int, AnswerRecord], None] | None = None,
    ) -> RunSummary:
        """기록이 없는 문항을 차례로 돌려 path 에 한 줄씩 덧붙인다.

        문항은 하나씩 순서대로 돈다 — 답변 서비스가 GPU 모델(검색 · Ollama)을 한 번에 하나씩 쓴다.

        Args:
          items: 평가 문항. id 가 겹치면 안 된다.
          path: 기록 JSONL. 없으면 만든다. 생성 실패(RETRY_ERRORS) 기록과 쓰다 만 마지막 줄은 지우고 다시 돈다.
          limit: 이번에 돌릴 최대 문항 수 (점검용). None 이면 남은 전부.
          on_record: 문항 하나를 기록할 때마다 (이번 순번, 이번에 돌릴 수, 기록) 으로 부른다 — 진행 표시.

        Raises:
          RuntimeError: 생성 실패가 MAX_CONSECUTIVE_ERRORS 번 이어졌을 때. 그때까지의 기록은 파일에 남는다.
        """
        if len({item.id for item in items}) != len(items):
            raise ValueError("item ids must be unique")
        rows, partial_line = _read_rows(path)
        kept = [row for row in rows if row["error"] not in RETRY_ERRORS]
        if partial_line or len(kept) != len(rows):
            _rewrite(path, kept)
        done = {row["item"]["id"] for row in kept}
        pending = [item for item in items if item.id not in done]
        todo = pending[:limit]
        streak = 0
        for index, item in enumerate(todo, start=1):
            record = await self.answer(item)
            _append(path, record)
            if on_record is not None:
                on_record(index, len(todo), record)
            streak = streak + 1 if record.error in RETRY_ERRORS else 0
            if streak >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"generation failed {streak} times in a row — check the LLM server and run again"
                )
        return RunSummary(
            total=len(items),
            skipped=len(items) - len(pending),
            recorded=len(todo),
            retried=len(rows) - len(kept),
        )

    async def answer(self, item: EvalItem) -> AnswerRecord:
        """문항 하나를 답변 경로로 돌려 기록을 만든다 (파일에 쓰지 않는다)."""
        if self._before_answer is not None:
            self._before_answer(item)
        self._search.last = None
        start = self._clock()
        events = [event async for event in self._answerer.answer(item.question)]
        wall_ms = (self._clock() - start) * 1000
        return build_record(item, events, self._search.last, wall_ms=wall_ms)


class _RecordingSearch:
    """검색기를 감싸 답변 경로가 받은 마지막 검색 결과를 남긴다."""

    def __init__(self, search: Searcher) -> None:
        self._search = search
        self.last: SearchResult | None = None

    async def search(self, query: str) -> SearchResult:
        self.last = await self._search.search(query)
        return self.last


def build_record(
    item: EvalItem, events: Sequence[Event], searched: SearchResult | None, *, wall_ms: float
) -> AnswerRecord:
    """답변 경로가 낸 이벤트와 그 경로가 받은 검색 결과로 기록 한 줄을 만든다.

    Args:
      item: 돌린 문항.
      events: 질문 하나의 SSE 이벤트 전부 (순서대로).
      searched: 답변 경로가 받은 검색 결과. 관문이 받아 검색하지 않았으면 None.
      wall_ms: 평가 쪽에서 잰 전체 시간 (ms).

    Raises:
      ValueError: 이벤트가 SSE 계약(ADR-027)과 맞지 않을 때 — done · error 가 하나가 아니거나, 출처 목록이 검색
        순서의 앞부분이 아닐 때 (출처 번호로 청크를 찾을 수 없다).
    """
    stages: list[str] = []
    pieces: list[str] = []
    sources: list[object] = []
    endings: list[Event] = []
    for event in events:
        if event.name == "stage":
            stages.append(str(event.data["stage"]))
        elif event.name == "sources":
            sources = _list(event.data["sources"])
        elif event.name == "delta":
            pieces.append(str(event.data["text"]))
        elif event.name in {"done", "error"}:
            endings.append(event)
    if len(endings) != 1:
        raise ValueError(
            f"expected exactly one done or error event, got {[e.name for e in events]}"
        )
    hits = tuple(
        ContextChunk(
            chunk_id=hit.chunk.id,
            title=hit.chunk.title,
            section=hit.chunk.section,
            text=hit.chunk.text,
            score=hit.score,
        )
        for hit in (searched.hits if searched is not None else ())
    )
    _check_sources(sources, hits)
    ending = endings[0]
    done = ending.data if ending.name == "done" else None
    diagnostics = _mapping(done["diagnostics"]) if done is not None else {}
    timings = _mapping(diagnostics.get("timings_ms", {}))
    gate = diagnostics.get("gate")
    return AnswerRecord(
        item=item,
        stage_path=tuple(stages),
        answer="".join(pieces),
        hits=hits,
        context_count=len(sources),
        marker=_optional_str(diagnostics.get("marker")),
        cited=_ints(done["cited"]) if done is not None else (),
        dropped=_ints(done["dropped"]) if done is not None else (),
        truncated=done is not None and done["truncated"] is True,
        timed_out=done is not None and done["timed_out"] is True,
        error=str(ending.data["code"]) if done is None else None,
        top_score=max((hit.score for hit in hits), default=None),
        gate_rule=str(_mapping(gate)["rule"]) if gate is not None else None,
        spec_id=_optional_str(diagnostics.get("spec_id")),
        prompt_version=_optional_str(diagnostics.get("prompt_version")),
        search_ms=_optional_float(timings.get("search")),
        first_delta_ms=_optional_float(timings.get("first_delta")),
        total_ms=_optional_float(timings.get("total")),
        wall_ms=round(wall_ms, 1),
        done_reason={
            str(stage): _optional_str(reason)
            for stage, reason in _mapping(diagnostics.get("done_reason", {})).items()
        },
    )


def tagged(path: Path, tag: str | None) -> Path:
    """다시 잰 기록의 경로 — 파일 이름 끝에 -{tag} (f9-answers-v1.jsonl → f9-answers-v1-grounded-v3.jsonl).

    tag 가 None 이면 path 그대로 (기준 기록). 기준 기록을 덮어쓰지 않으려고 실행 스크립트들이 같이 쓴다.
    """
    return path if tag is None else path.with_name(f"{path.stem}-{tag}{path.suffix}")


def read_records(path: Path) -> list[AnswerRecord]:
    """f9-answers JSONL 의 기록. 쓰다 만 마지막 줄(실행이 끊김)은 버린다."""
    rows, _ = _read_rows(path)
    return [AnswerRecord.from_json(row) for row in rows]


def _check_sources(sources: Sequence[object], hits: Sequence[ContextChunk]) -> None:
    # 답변 정책은 검색 순서(리랭커 점수 내림차순)에서 low 미만만 빼므로 출처는 검색 결과의 앞부분이다
    if len(sources) > len(hits):
        raise ValueError(f"{len(sources)} sources but only {len(hits)} search hits")
    for number, (source, hit) in enumerate(zip(sources, hits, strict=False), start=1):
        fields = _mapping(source)
        if fields["number"] != number or fields["title"] != hit.title:
            raise ValueError(
                f"source {number} does not follow the search order: "
                f"{fields['number']!r} {fields['title']!r} vs {hit.title!r}"
            )


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """JSONL 의 줄들과, 쓰다 만 마지막 줄을 버렸는지."""
    if not path.exists():
        return [], False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index < len(lines) - 1:
                raise
            logger.warning("dropping a partial last line in %s", path)
            return rows, True
    return rows, False


def _rewrite(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    # 임시 파일에 다 쓴 뒤 바꿔 끼운다 — 도중에 끊겨도 원래 기록이 남는다
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append(path: Path, record: AnswerRecord) -> None:
    # 문항마다 열고 닫는다 — 실행이 끊겨도 끝난 문항은 파일에 남는다
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {value!r}")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping, got {value!r}")
    return value


def _ints(value: object) -> tuple[int, ...]:
    return tuple(cast("list[int]", _list(value)))


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(cast("float", value))
