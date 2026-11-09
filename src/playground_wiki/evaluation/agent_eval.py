"""단발 대 에이전트 비교 측정 (scripts/eval_agent.py): 같은 세트를 두 모드로 돌린 기록 · 집계 · 리포트.

evaluation 은 api 를 import 하지 않는다 — 답변 서비스는 `ModeAnswerer` 계약으로 받는다. 검색기를 감싸 검색 횟수 ·
검색어 · 찾은 문서 제목을 기록에 남긴다 (done 진단에는 하위 질문 · 검색어를 싣지 않아서 — ADR-027).
기록은 로컬(data/)에만 두고, 리포트에는 집계와 문항 id 만 쓴다 (KorQuAD 원문 금지).
"""

import functools
import itertools
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, get_args

from playground_evalkit import mcnemar_exact, normalize_title, percentile, render_table, squash
from playground_retrieval.core import Hit

from playground_wiki.evaluation.answer_metrics import (
    EXPECTED_FINAL,
    LATENCY_POINTS,
    Share,
    format_counts,
    format_share,
)
from playground_wiki.evaluation.answer_run import (
    ContextChunk,
    EvalItem,
    Event,
    RunSummary,
    ending_error,
    run_pending,
)
from playground_wiki.evaluation.files import (
    as_mapping,
    optional_str,
    read_jsonl_tail,
)
from playground_wiki.evaluation.search_adapters import RecordingSearch, SearchCall

CompareSet = Literal["compound", "f9", "offwiki"]
"""맞힘을 견주는 세트 (`render_report`). compound 는 복합 질문 50 (tests/evaluation/compound-v1.jsonl), f9 · offwiki 는
F9 표본의 KorQuAD · 위키 밖 문항."""
COMPARE_SETS: tuple[str, ...] = get_args(CompareSet)
SafetySet = Literal["harmful", "procedure", "f4"]
"""안전 회귀 세트 (evaluation.agent_safety). harmful 은 유해 정보 20 + 대조 20, procedure 는 무해 절차 20, f4 는 F4 저장 문맥
문항 (반사실 · 민감 주제 · 인젝션 · 답 있음 대조군 · 답 없음)."""
SAFETY_SETS: tuple[str, ...] = get_args(SafetySet)
SetName = Literal[CompareSet, SafetySet]
SET_NAMES: tuple[str, ...] = get_args(SetName)
Mode = Literal["single", "agent"]
"""답변 모드 — /ask 의 mode 와 같은 값."""
MODES: tuple[str, ...] = get_args(Mode)
AGENT_STAGE_PREFIX = "agent_"
"""에이전트 노드의 stage 이벤트 앞말 (api.answer 와 같은 값 — evaluation 은 api 를 import 하지 않아 여기 적는다)."""
TOP_PATHS = 6  # 리포트에 보일 에이전트 단계 경로 가짓수


class ModeAnswerer(Protocol):
    """질문 하나에 SSE 이벤트를 흘린다 (api.answer.AnswerService)."""

    def answer(self, question: str, *, mode: Mode = "single") -> AsyncIterator[Event]:
        """질문 하나의 이벤트 (stage … done 또는 error)."""
        ...


@dataclass(frozen=True)
class AgentItem:
    """측정 문항 하나."""

    id: str
    set_name: SetName
    question: str
    answers: tuple[str, ...] = ()
    """정답 표기 (공백 무시 문자열 포함으로 맞힘). 위키 밖 질문은 비어 있다."""
    expected: str | None = None
    """위키 밖 질문의 기대 단계 (partial · refuse)."""
    kind: str | None = None
    """복합 질문의 꼴 (bridge · comparison …), 위키 밖 질문의 종류."""
    hop_titles: tuple[str, ...] = ()
    """복합 질문의 hop 문서 제목 (정규화한 것, 계산 hop 은 빠진다)."""

    @classmethod
    def from_compound(cls, row: Mapping[str, Any]) -> Self:
        """compound-v1.jsonl 한 줄에서."""
        titles = [str(hop["title"]) for hop in row["hops"] if hop["title"] is not None]
        return cls(
            id=str(row["id"]),
            set_name="compound",
            question=str(row["question"]),
            answers=(str(row["answer"]), *(str(alias) for alias in row["aliases"])),
            kind=str(row["kind"]),
            hop_titles=tuple(dict.fromkeys(normalize_title(title) for title in titles)),
        )

    @classmethod
    def from_eval_item(cls, item: EvalItem) -> Self:
        """F9 표본 문항(`EvalItem`)에서 — korquad 는 f9, offwiki 는 offwiki 세트."""
        return cls(
            id=item.id,
            set_name="offwiki" if item.source == "offwiki" else "f9",
            question=item.question,
            answers=item.gold_answers,
            expected=item.expected,
            kind=item.category,
        )

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """기록의 item 칸에서."""
        return cls(
            **{
                **row,
                "answers": tuple(row["answers"]),
                "hop_titles": tuple(row["hop_titles"]),
            }
        )


def compound_items(rows: Iterable[Mapping[str, Any]]) -> list[AgentItem]:
    """복합 질문 세트의 문항들을 돌려준다."""
    return [AgentItem.from_compound(row) for row in rows]


def f9_items(rows: Iterable[Mapping[str, Any]], set_name: SetName) -> list[AgentItem]:
    """F9 표본 줄에서 set_name(f9 = KorQuAD · offwiki = 위키 밖) 문항만 돌려준다."""
    source = "offwiki" if set_name == "offwiki" else "korquad"
    return [
        AgentItem.from_eval_item(EvalItem.from_json(row)) for row in rows if row["source"] == source
    ]


@dataclass(frozen=True)
class AgentRecord:
    """문항 하나를 한 모드로 돌린 기록. 기록 JSONL 한 줄."""

    item: AgentItem
    mode: Mode
    stage_path: tuple[str, ...]
    """stage 이벤트 순서 그대로 (에이전트 노드 agent_* 포함)."""
    answer: str
    """delta 를 이어 붙인 답 (고정 문구 포함)."""
    source_titles: tuple[str, ...]
    """sources 이벤트의 문서 제목 — 답변 근거로 넘긴 청크 (번호 순서)."""
    searches: tuple[SearchCall, ...]
    error: str | None
    """실패 코드 (`answer_run.ending_error`). 끝까지 답했으면 None."""
    first_delta_ms: float | None
    total_ms: float | None
    wall_ms: float
    """평가 쪽에서 잰 전체 시간 (ms)."""
    gate_rule: str | None = None
    top_score: float | None = None
    agent: Mapping[str, Any] = field(default_factory=dict)
    """done 진단의 agent 칸 (단발이면 비어 있다)."""
    marker: str | None = None
    """done 진단의 grounded 첫 줄 표지 (found · not_found · missing). grounded 를 거치지 않았거나 이 칸이 생기기 전
    기록이면 None."""
    context: tuple[ContextChunk, ...] = ()
    """답변 근거로 넘긴 청크 본문 (sources 차례 — 출처 번호 n = context[n - 1]). 사람 검수표 · F4 채점이 읽는다.
    이 칸이 생기기 전 기록이면 비어 있다. 기록은 로컬에만 둔다."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", MappingProxyType(dict(self.agent)))

    @property
    def answer_path(self) -> tuple[str, ...]:
        """답변 단계만 (관문 · grounded · partial · refuse) — 에이전트 노드를 뺀 것."""
        return tuple(stage for stage in self.stage_path if not stage.startswith(AGENT_STAGE_PREFIX))

    @property
    def final_stage(self) -> str:
        """끝난 단계. 실패면 error, partial 이 답 조각 없이 고정 문구로 끝났으면 partial_empty (answer_metrics 와 같은 규칙)."""
        path = self.answer_path
        if self.error is not None or not path:
            return "error"
        if path[-1] == "partial" and self.first_delta_ms is None:
            return "partial_empty"
        return path[-1]

    @property
    def iterations(self) -> int:
        """검색 반복 수. 단발은 검색했으면 1. 에이전트는 그래프 반복 수 — route 가 있으면(단발 먼저) 원문 검색 1 을 더한다."""
        if self.mode != "agent":
            return 1 if self.searches else 0
        graph = int(self.agent.get("iterations", 0))
        if self.route is None:  # route 가 생기기 전 기록 (에이전트가 바로 검색)
            return graph
        return graph + (1 if self.route != "gate" else 0)

    @property
    def route(self) -> str | None:
        """에이전트 모드가 간 길 (done 진단 agent.route). 단발이거나 route 가 생기기 전 기록이면 None."""
        value = self.agent.get("route")
        return None if value is None else str(value)

    def to_json(self) -> dict[str, Any]:
        """JSONL 한 줄로 쓸 값."""
        return {
            "item": asdict(self.item),
            "mode": self.mode,
            "stage_path": list(self.stage_path),
            "answer": self.answer,
            "source_titles": list(self.source_titles),
            "searches": [asdict(call) for call in self.searches],
            "error": self.error,
            "first_delta_ms": self.first_delta_ms,
            "total_ms": self.total_ms,
            "wall_ms": self.wall_ms,
            "gate_rule": self.gate_rule,
            "top_score": self.top_score,
            "agent": dict(self.agent),
            "marker": self.marker,
            "context": [asdict(chunk) for chunk in self.context],
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """기록 JSONL 한 줄에서. marker · context 가 없는 예전 기록은 빈 값."""
        return cls(
            **{
                **row,
                "item": AgentItem.from_json(row["item"]),
                "stage_path": tuple(row["stage_path"]),
                "source_titles": tuple(row["source_titles"]),
                "searches": tuple(
                    SearchCall(call["query"], tuple(call["titles"])) for call in row["searches"]
                ),
                "context": tuple(ContextChunk(**chunk) for chunk in row.get("context", ())),
            }
        )


def build_record(
    item: AgentItem,
    mode: Mode,
    events: Sequence[Event],
    searches: Sequence[SearchCall],
    *,
    wall_ms: float,
    seen: Sequence[Hit] = (),
) -> AgentRecord:
    """이벤트와 기록한 검색으로 기록 한 줄을 만든다.

    Args:
      item: 돌린 문항.
      mode: 답변 모드.
      events: 질문 하나의 SSE 이벤트 전부.
      searches: 기록한 검색 (검색어 · 찾은 문서 제목).
      wall_ms: 평가 쪽에서 잰 전체 시간 (ms).
      seen: 검색이 돌려준 청크 전부 (검색 차례). 마지막 sources 의 출처를 이 청크와 (제목, 섹션, 점수)로 맞춰 답변 근거
        본문(context)을 남긴다 — 에이전트 답변 근거는 여러 검색의 청크를 섞은 것이라 검색 결과 하나의 앞부분이 아니다.
        비었으면 context 도 비운다.

    Raises:
      ValueError: done · error 이벤트가 하나가 아닐 때 (SSE 계약 위반), seen 이 있는데 출처가 그 안에 없을 때.
    """
    stages: list[str] = []
    pieces: list[str] = []
    rows: list[Mapping[str, Any]] = []
    endings = [event for event in events if event.name in {"done", "error"}]
    if len(endings) != 1:
        raise ValueError(f"expected one done or error event, got {[e.name for e in events]}")
    for event in events:
        if event.name == "stage":
            stages.append(str(event.data["stage"]))
        elif event.name == "delta":
            pieces.append(str(event.data["text"]))
        elif event.name == "sources":
            rows = _rows(event.data["sources"])
    titles = [str(source["title"]) for source in rows]
    ending = endings[0]
    diagnostics = as_mapping(ending.data["diagnostics"]) if ending.name == "done" else {}
    timings = as_mapping(diagnostics.get("timings_ms", {}))
    gate = diagnostics.get("gate")
    return AgentRecord(
        item=item,
        mode=mode,
        stage_path=tuple(stages),
        answer="".join(pieces),
        source_titles=tuple(titles),
        searches=tuple(searches),
        error=ending_error(ending),
        first_delta_ms=_number(timings.get("first_delta")),
        total_ms=_number(timings.get("total")),
        wall_ms=round(wall_ms, 1),
        gate_rule=str(as_mapping(gate)["rule"]) if gate is not None else None,
        top_score=_number(diagnostics.get("top_score")),
        agent=as_mapping(diagnostics.get("agent", {})),
        marker=optional_str(diagnostics.get("marker")),
        context=_context(rows, seen) if seen else (),
    )


def _context(sources: Sequence[Mapping[str, Any]], seen: Sequence[Hit]) -> tuple[ContextChunk, ...]:
    """출처마다 (제목, 섹션, 점수)가 같은 첫 검색 청크의 본문. 답변 경로는 검색이 돌려준 Hit 를 그대로 근거로 넘긴다."""
    by_key: dict[tuple[str, str, float], ContextChunk] = {}
    for hit in seen:
        key = (hit.chunk.title, hit.chunk.section, hit.score)
        by_key.setdefault(
            key,
            ContextChunk(
                chunk_id=hit.chunk.id,
                title=hit.chunk.title,
                section=hit.chunk.section,
                text=hit.chunk.text,
                score=hit.score,
            ),
        )
    chunks = []
    for source in sources:
        key = (str(source["title"]), str(source["section"]), float(source["score"]))
        if key not in by_key:
            raise ValueError(f"source {source['number']} is not among the searched chunks")
        chunks.append(by_key[key])
    return tuple(chunks)


class AgentEvalRun:
    """문항을 한 모드로 돌려 문항마다 기록 한 줄을 이어 쓴다. 기록이 있는 문항은 건너뛴다.

    예)
        recording = RecordingSearch(search_service)
        run = AgentEvalRun(recording, AnswerService.from_config(setup, search=recording, llm=llm))
        asyncio.run(run.run(items, "agent", Path("data/eval/agent/compound-agent.jsonl")))
    """

    def __init__(
        self,
        search: RecordingSearch,
        answerer: ModeAnswerer,
        *,
        before_answer: Callable[[AgentItem], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._search = search
        self._answerer = answerer
        # 문항마다 답변 경로를 부르기 직전에 — 문항의 저장 문맥을 돌려주는 검색기(F4)처럼 바꿔 끼울 것이 있을 때
        self._before_answer = before_answer
        self._clock = clock

    async def run(
        self,
        items: Sequence[AgentItem],
        mode: Mode,
        path: Path,
        *,
        limit: int | None = None,
        on_record: Callable[[int, int, AgentRecord], None] | None = None,
    ) -> RunSummary:
        """기록이 없는 문항을 차례로 돌려 path 에 한 줄씩 덧붙인다 (GPU 를 한 번에 하나씩 쓰게 순서대로).

        Args:
          items: 측정 문항 (id 가 겹치면 안 된다).
          mode: 답변 모드.
          path: 기록 JSONL. 생성 실패 기록과 쓰다 만 마지막 줄은 지우고 다시 돈다.
          limit: 이번에 돌릴 최대 문항 수. None 이면 남은 전부.
          on_record: 문항 하나를 기록할 때마다 (이번 순번, 이번에 돌릴 수, 기록).

        Raises:
          RuntimeError: 생성 실패가 MAX_CONSECUTIVE_ERRORS 번 이어졌을 때 (그때까지의 기록은 남는다).
        """
        return await run_pending(
            items,
            path,
            functools.partial(self.answer, mode=mode),
            limit=limit,
            on_record=on_record,
        )

    async def answer(self, item: AgentItem, mode: Mode) -> AgentRecord:
        """문항 하나를 답변 경로로 돌려 기록을 만든다 (파일에 쓰지 않는다)."""
        if self._before_answer is not None:
            self._before_answer(item)
        self._search.reset()
        start = self._clock()
        events = [event async for event in self._answerer.answer(item.question, mode=mode)]
        wall_ms = (self._clock() - start) * 1000
        return build_record(
            item,
            mode,
            events,
            list(self._search.calls),
            wall_ms=wall_ms,
            seen=list(self._search.hits),
        )


def read_records(path: Path) -> list[AgentRecord]:
    """기록 JSONL 의 기록 (쓰다 만 마지막 줄은 버린다). 파일이 없으면 빈 목록."""
    rows, _ = read_jsonl_tail(path)
    return [AgentRecord.from_json(row) for row in rows]


def contains_answer(record: AgentRecord) -> bool | None:
    """답에 정답 표기가 하나라도 있나 (공백 무시 — answer_metrics.contains_gold 와 같은 규칙). 정답이 없으면 None."""
    if not record.item.answers:
        return None
    text = squash(record.answer)
    return any(squash(answer) in text for answer in record.item.answers)


def meets_expected(record: AgentRecord) -> bool | None:
    """위키 밖 질문이 기대 단계로 끝났나 (answer_metrics.matches_expected 와 같은 규칙). 기대 단계가 없으면 None."""
    if record.item.expected is None:
        return None
    return record.final_stage in EXPECTED_FINAL[record.item.expected]


def correct(record: AgentRecord) -> bool | None:
    """세트의 맞힘 기준 — 정답이 있으면 정답 포함, 위키 밖 질문이면 기대 단계."""
    return contains_answer(record) if record.item.answers else meets_expected(record)


def hop_coverage(record: AgentRecord, *, searched: bool = False) -> tuple[int, int]:
    """복합 질문의 hop 문서 중 덮은 수와 전체 수. searched 면 검색한 문서 전부, 아니면 답변 근거(sources)."""
    titles = record.source_titles
    if searched:
        titles = tuple(itertools.chain.from_iterable(call.titles for call in record.searches))
    found = {normalize_title(title) for title in titles}
    hops = record.item.hop_titles
    return sum(title in found for title in hops), len(hops)


@dataclass(frozen=True)
class Pairing:
    """같은 세트의 두 모드 기록을 문항 id 로 짝지은 것 (두 모드 모두 기록이 있는 문항만)."""

    set_name: str
    single: tuple[AgentRecord, ...]
    agent: tuple[AgentRecord, ...]
    """single 과 같은 순서 (문항 id 로 맞춘다)."""

    @classmethod
    def match(
        cls, set_name: str, single: Sequence[AgentRecord], agent: Sequence[AgentRecord]
    ) -> Self:
        """두 기록 목록에서 두 모드 모두 있는 문항만 single 순서로 짝짓는다."""
        by_id = {record.item.id: record for record in agent}
        paired = [(record, by_id[record.item.id]) for record in single if record.item.id in by_id]
        return cls(
            set_name=set_name,
            single=tuple(pair[0] for pair in paired),
            agent=tuple(pair[1] for pair in paired),
        )


def render_report(pairings: Sequence[Pairing], *, today: date, settings: Mapping[str, str]) -> str:
    """세트마다 단발 대 에이전트 비교 리포트 (markdown). 문항 원문 없이 집계와 id 만."""
    sections = [
        f"# 단발 대 에이전트 비교 ({today:%Y-%m-%d})\n",
        "같은 문항을 `/ask` 단발(mode=single)과 에이전트(mode=agent)로 돌린 기록을 문항 id 로 짝지었다. "
        "맞힘은 정답이 있는 세트(복합 · F9 KorQuAD)에서 답에 정답 표기가 있는지(공백 무시), 위키 밖 세트에서 "
        "기대 단계(partial · refuse)로 끝났는지. 검정은 McNemar 정확 검정(엇갈린 문항만).\n",
        _settings_section(settings),
    ]
    sections.extend(_set_section(pairing) for pairing in pairings if pairing.single)
    return "\n".join(sections)


def _settings_section(settings: Mapping[str, str]) -> str:
    rows = [{"항목": key, "값": f"`{value}`"} for key, value in settings.items()]
    return "## 설정\n\n" + render_table(rows, ["항목", "값"]) if rows else ""


def _set_section(pairing: Pairing) -> str:
    single, agent = pairing.single, pairing.agent
    flags_single = [correct(record) for record in single]
    flags_agent = [correct(record) for record in agent]
    usable = [
        (a, b)
        for a, b in zip(flags_single, flags_agent, strict=True)
        if a is not None and b is not None
    ]
    parts = [f"## {pairing.set_name} ({len(single)} 문항)\n"]
    parts.append(_overview_table(single, agent))
    if usable:
        parts.append(_paired_table(pairing, usable))
    parts.append(_stage_table(single, agent))
    parts.append(_agent_paths(agent))
    if any(record.item.hop_titles for record in single):
        parts.append(_hop_table(single, agent))
        parts.append(_kind_table(single, agent))
    return "\n".join(parts)


def _overview_table(single: Sequence[AgentRecord], agent: Sequence[AgentRecord]) -> str:
    rows = [_overview_row("단발", single), _overview_row("에이전트", agent)]
    columns = [
        "모드",
        "맞힘",
        "평균 검색 횟수",
        "평균 반복",
        "지연 p50 (s)",
        "지연 p95 (s)",
        "첫 글자 p50 (s)",
        "실패",
    ]
    return "### 요약\n\n" + render_table(rows, columns)


def _overview_row(name: str, records: Sequence[AgentRecord]) -> dict[str, str | int | float]:
    flags = [flag for flag in (correct(record) for record in records) if flag is not None]
    walls = [record.wall_ms / 1000 for record in records]
    firsts = [record.first_delta_ms / 1000 for record in records if record.first_delta_ms]
    return {
        "모드": name,
        "맞힘": format_share(Share(sum(flags), len(flags))),
        "평균 검색 횟수": _mean([len(record.searches) for record in records]),
        "평균 반복": _mean([record.iterations for record in records]),
        "지연 p50 (s)": _point(walls, LATENCY_POINTS[0]),
        "지연 p95 (s)": _point(walls, LATENCY_POINTS[1]),
        "첫 글자 p50 (s)": _point(firsts, LATENCY_POINTS[0]),
        "실패": sum(record.error is not None for record in records),
    }


def _paired_table(pairing: Pairing, usable: Sequence[tuple[bool, bool]]) -> str:
    test = mcnemar_exact([a for a, _ in usable], [b for _, b in usable])
    both = sum(a and b for a, b in usable)
    neither = sum(not a and not b for a, b in usable)
    row = {
        "짝": len(usable),
        "둘 다": both,
        "단발만": test.only_a,
        "에이전트만": test.only_b,
        "둘 다 틀림": neither,
        "McNemar p": f"{test.p_value:.3g}",
    }
    columns = ["짝", "둘 다", "단발만", "에이전트만", "둘 다 틀림", "McNemar p"]
    only_single = [
        s.item.id
        for s, a in zip(pairing.single, pairing.agent, strict=True)
        if correct(s) and not correct(a)
    ]
    only_agent = [
        s.item.id
        for s, a in zip(pairing.single, pairing.agent, strict=True)
        if correct(a) and not correct(s)
    ]
    return (
        "### 같은 문항 짝\n\n"
        + render_table([row], columns)
        + f"\n- 단발만 맞힘: {', '.join(only_single) or '없음'}\n"
        + f"- 에이전트만 맞힘: {', '.join(only_agent) or '없음'}\n"
    )


def _stage_table(single: Sequence[AgentRecord], agent: Sequence[AgentRecord]) -> str:
    counts = {
        name: Counter(record.final_stage for record in records)
        for name, records in (("단발", single), ("에이전트", agent))
    }
    stages = sorted(set(counts["단발"]) | set(counts["에이전트"]))
    rows = [
        {"모드": name, **{stage: counter[stage] for stage in stages}}
        for name, counter in counts.items()
    ]
    return "### 끝난 단계\n\n" + render_table(rows, ["모드", *stages])


def _agent_paths(agent: Sequence[AgentRecord]) -> str:
    paths = Counter(" → ".join(record.agent.get("steps", ())) or "(관문)" for record in agent)
    routes = Counter(record.route or "(route 없음)" for record in agent)
    ran = [record for record in agent if record.agent.get("steps")]
    stops = Counter(str(record.agent.get("stop")) for record in ran)
    decompose = Counter(str(record.agent.get("decompose")) for record in ran)
    rows = [{"단계 경로": path, "문항": count} for path, count in paths.most_common(TOP_PATHS)]
    return (
        "### 에이전트 단계 경로\n\n"
        + render_table(rows, ["단계 경로", "문항"])
        + f"\n- 길(route): {format_counts(routes)}\n"
        + f"- 그래프를 돈 문항 {len(ran)} — 멈춘 이유: {format_counts(stops)} · 질문 나누기: {format_counts(decompose)}\n"
    )


def _hop_table(single: Sequence[AgentRecord], agent: Sequence[AgentRecord]) -> str:
    rows = []
    for name, records in (("단발", single), ("에이전트", agent)):
        for label, searched in (("답변 근거", False), ("검색한 문서", True)):
            pairs = [hop_coverage(record, searched=searched) for record in records]
            full = sum(found == total for found, total in pairs if total)
            found_sum, total_sum = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
            rows.append(
                {
                    "모드": name,
                    "대상": label,
                    "hop 모두 덮음": format_share(
                        Share(full, sum(1 for _, total in pairs if total))
                    ),
                    "hop 덮은 비율": format_share(Share(found_sum, total_sum)),
                }
            )
    return "### hop 문서 덮음\n\n" + render_table(
        rows, ["모드", "대상", "hop 모두 덮음", "hop 덮은 비율"]
    )


def _kind_table(single: Sequence[AgentRecord], agent: Sequence[AgentRecord]) -> str:
    kinds = sorted({str(record.item.kind) for record in single})
    rows = []
    for kind in kinds:
        row: dict[str, str | int | float] = {"꼴": kind}
        for name, records in (("단발", single), ("에이전트", agent)):
            flags = [bool(correct(r)) for r in records if r.item.kind == kind]
            row[name] = format_share(Share(sum(flags), len(flags)))
        rows.append(row)
    return "### 꼴별 맞힘\n\n" + render_table(rows, ["꼴", "단발", "에이전트"])


def _mean(values: Sequence[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "-"


def _point(values: Sequence[float], q: int) -> str:
    return f"{percentile(values, q):.1f}" if values else "-"


def _rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {value!r}")
    return [as_mapping(row) for row in value]


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
