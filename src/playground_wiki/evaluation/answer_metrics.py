"""F9 답변 평가의 자동 지표 — 답변 기록(`AnswerRecord`)에서 사람 · 판정 모델 없이 셀 수 있는 것만.

정답 포함은 KorQuAD 정답 문자열이 답에 있는지다 (공백 무시 — evalkit `contains_answer` 와 같은 규칙). 답이 문장이라
EM · F1 은 쓰지 않는다 (짧은 답을 따로 받는 건 다음 일). 문항 하나를 보는 판정 함수는 그 문항이 분모에 들지 않으면
None 을 돌려주고, `share` 가 None 을 분모에서 뺀다.
"""

import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast, get_args

from playground_evalkit import KorQuADItem, contains_answer, mcnemar_exact, percentile, squash
from playground_retrieval.core import Hit

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk

ANSWER_STAGES = frozenset({"grounded", "partial"})
"""LLM 이 답을 쓰는 단계 (SSE stage 값, ADR-027 — evaluation 은 generation 을 import 하지 않아 이름을 여기 적는다)."""
FINAL_STAGES: tuple[str, ...] = ("gate", "refuse", "grounded", "partial", "partial_empty", "error")
"""마지막 단계 표의 열 순서. partial_empty 는 partial 이 생성 조각을 하나도 못 보내 고정 거절 문구로 끝난 것,
error 는 답을 하나도 못 보낸 채 실패한 것."""
FIXED_REFUSAL = "고정 거절"
"""partial_empty 를 단계 경로 끝에 붙이는 이름."""
EXPECTED_FINAL: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "refuse": frozenset({"gate", "refuse", "partial_empty"}),
        "partial": frozenset({"partial"}),
    }
)
"""위키 밖 질문의 기대 단계 → 기대대로 끝난 것으로 치는 마지막 단계. 관문의 고정 답 · 고정 거절 문구도 거절 경로다."""
MarkerOutcome = Literal["found", "not_found", "missing", "cut_before_marker", "empty"]
MARKER_OUTCOMES: tuple[str, ...] = get_args(MarkerOutcome)
SCORE_BAND_EDGES = (0.83, 0.94)
"""리랭커 1위 점수 구간 경계 — 0.83~0.94 구간의 grounded 를 0.94 이상과 견줘 high 를 정한다 (ADR-024)."""
MARKER_MISSING_LIMIT = 0.01
"""표지 누락률이 이 미만이면 "표지 없으면 찾음" 기본값을 유지한다 (ADR-024)."""
LATENCY_PERCENTILES = (50, 90, 95, 99)
# '찾음' 표지 뒤 본문이 문서에 답이 없다고 말하는 꼴 (ADR-027 스모크: "…에 대한 정보는 없습니다"). 정규식 조각을 | 로 잇는다.
# "최근 값은 알 수 없다"(base@v4 가 시킨 시점 안내)는 값과 함께 나오는 정상 답이라 넣지 않는다.
# "포함되지 않는다" 는 사실 답("명왕성은 행성에 포함되지 않는다")에도 흔해서 넣지 않는다
NO_INFO_PATTERNS: tuple[str, ...] = (
    r"(?:정보|내용|언급|기록|자료)[이가은는도]?\s*(?:전혀\s*)?없",
    r"찾을\s*수\s*(?:가\s*)?없",
    r"(?:언급|명시|기재|기술)(?:되어|돼)\s*있지\s*않",
    r"(?:언급|명시|기재|기술|제공)되지\s*않",
    r"나와\s*있지\s*않",
    r"(?:문서|자료)에서는?\s*확인(?:할\s*수\s*없|되지\s*않)",
)
_NO_INFO = re.compile("|".join(NO_INFO_PATTERNS))


@dataclass(frozen=True)
class Share:
    """비율 하나: 해당 문항 수 / 분모 문항 수."""

    count: int
    total: int

    @property
    def rate(self) -> float | None:
        """비율 (count / total). 분모가 0 이면 None."""
        return self.count / self.total if self.total else None


@dataclass(frozen=True)
class Latency:
    """지연 백분위 — 값이 있는 문항만."""

    count: int
    percentiles: Mapping[int, float]
    """백분위(LATENCY_PERCENTILES) → ms. 값이 없으면 비어 있다."""


@dataclass(frozen=True)
class PairedHits:
    """같은 문항을 원문 · 패러프레이즈로 검색한 성공 여부(hit@k)를 짝지은 표."""

    both: int
    original_only: int
    paraphrase_only: int
    neither: int
    mcnemar_p: float
    """두 hit@k 가 같다는 가설의 정확 McNemar p 값 (엇갈린 짝만 보는 양측 검정, evalkit `mcnemar_exact`)."""

    @property
    def total(self) -> int:
        """짝지은 문항 수."""
        return self.both + self.original_only + self.paraphrase_only + self.neither

    @property
    def original_rate(self) -> float | None:
        """원문 질문의 hit@k. 문항이 없으면 None."""
        return (self.both + self.original_only) / self.total if self.total else None

    @property
    def paraphrase_rate(self) -> float | None:
        """패러프레이즈의 hit@k. 문항이 없으면 None."""
        return (self.both + self.paraphrase_only) / self.total if self.total else None


def share(flags: Iterable[bool | None]) -> Share:
    """True 개수 / None 을 뺀 개수."""
    counted = [flag for flag in flags if flag is not None]
    return Share(count=sum(counted), total=len(counted))


def final_stage(record: AnswerRecord) -> str:
    """기록이 끝난 단계 (FINAL_STAGES 중 하나). 답을 하나도 못 보낸 채 실패했으면 error.

    partial 로 끝났는데 첫 답 조각 시간(done 진단의 timings_ms.first_delta)이 없으면 partial_empty — 생성 조각을 하나도
    보내지 않고 고정 거절 문구만 나갔다 (/ask 는 고정 문구를 첫 답 조각 시간에 세지 않는다). 이때 done 의 truncated 는
    False 라 잘림으로는 가를 수 없고, 한도에 닿지 않고 빈 답으로 끝난 것(stop)도 같은 고정 문구다.
    """
    if record.error is not None or not record.stage_path:
        return "error"
    if record.stage_path[-1] == "partial" and record.first_delta_ms is None:
        return "partial_empty"
    return record.stage_path[-1]


def stage_path(record: AnswerRecord) -> str:
    """단계 경로 이름 ("grounded → partial"). 실패로 끝났으면 끝에 error, 고정 거절 문구로 끝났으면 FIXED_REFUSAL."""
    ending = {"error": ["error"], "partial_empty": [FIXED_REFUSAL]}.get(final_stage(record), [])
    return " → ".join([*record.stage_path, *ending])


def path_counts(records: Iterable[AnswerRecord]) -> dict[str, int]:
    """단계 경로별 문항 수. 많은 순, 같으면 이름순."""
    counts = Counter(stage_path(record) for record in records)
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def stage_counts(records: Iterable[AnswerRecord]) -> dict[str, int]:
    """끝난 단계별 문항 수 (FINAL_STAGES 순서, 없는 단계는 0)."""
    counts = Counter(final_stage(record) for record in records)
    return {stage: counts[stage] for stage in FINAL_STAGES}


def answered(record: AnswerRecord) -> bool:
    """LLM 단계(grounded · partial)가 쓴 답으로 끝났나. 고정 거절 문구로 끝난 partial(partial_empty)은 아니다."""
    return final_stage(record) in ANSWER_STAGES


def contains_gold(record: AnswerRecord) -> bool | None:
    """답에 정답 문자열(표기 중 하나라도)이 있나 (공백 무시). 정답이 없는 문항(위키 밖)이면 None."""
    if record.item.gold_answer is None:
        return None
    text = squash(record.answer)
    return any(squash(answer) in text for answer in record.item.gold_answers)


def gold_numbers(record: AnswerRecord) -> tuple[int, ...]:
    """문맥으로 넘긴 청크 중 정답 근거(정답 문서이고 정답 문자열을 담음)의 출처 번호. 정답이 없는 문항이면 빈 튜플."""
    items = record.item.korquad_items()
    context = record.hits[: record.context_count]
    return tuple(
        number for number, chunk in enumerate(context, start=1) if _is_gold_chunk(chunk, items)
    )


def gold_in_hits(record: AnswerRecord) -> bool | None:
    """답변 경로의 검색 결과(문맥으로 넘기지 않은 것 포함)에 정답 근거 청크가 있나. 정답이 없는 문항이면 None."""
    items = record.item.korquad_items()
    if not items:
        return None
    return any(_is_gold_chunk(chunk, items) for chunk in record.hits)


def _is_gold_chunk(chunk: ContextChunk, items: Sequence[KorQuADItem]) -> bool:
    # 정답 표기 중 하나라도 — 롱테일 세트는 표기가 여럿이다 (KorQuAD 는 하나)
    return any(contains_answer(chunk.title, chunk.text, item) for item in items)


def cites_gold(record: AnswerRecord) -> bool | None:
    """답이 정답 근거 청크를 출처 번호로 짚었나. 정답 문항이 아니거나 문맥에 정답 근거 청크가 없으면 None."""
    numbers = gold_numbers(record)
    if not numbers:
        return None
    return any(number in numbers for number in record.cited)


def says_no_info(text: str) -> bool:
    """글이 '문서에 정보가 없다 · 찾을 수 없다' 꼴의 말을 하나 (NO_INFO_PATTERNS)."""
    return _NO_INFO.search(text) is not None


def marker_outcome(record: AnswerRecord) -> MarkerOutcome | None:
    """첫 줄 표지가 어떻게 끝났나 — grounded 출력 첫머리 (MARKER_OUTCOMES 중 하나).

    found · not_found 는 표지를 쓴 것, missing 은 표지 없이 답을 써 찾음으로 흘린 것(ADR-024 의 표지 누락)이다.
    표지 없이 아무것도 못 보내고 partial 로 넘어간 것은 누락이 아니라 따로 센다 — grounded 의 done_reason 이 length 면
    cut_before_marker(표지를 정하기 전에 max_tokens 에 닿음), 그 밖이면 empty(빈 답). done_reason 이 없는 예전 기록은
    표지 전 잘림이 partial 로 넘어가지 않던 때라 empty 다. 표지를 읽지 않은 기록(grounded 를 안 거침 · 표지 전에 시간
    초과 · 실패)이면 None.
    """
    if record.marker is None:
        return None
    if record.marker != "missing" or _grounded_sent(record):
        return cast("MarkerOutcome", record.marker)
    return "cut_before_marker" if record.done_reason.get("grounded") == "length" else "empty"


def marker_missing(record: AnswerRecord) -> bool | None:
    """표지 없이 답을 썼나 (ADR-024 의 표지 누락).

    표지를 쓰거나 표지 없이 답을 쓴 기록만 센다 — 표지 전 잘림 · 빈 답 · 표지를 읽지 않은 기록이면 None.
    """
    outcome = marker_outcome(record)
    if outcome not in {"found", "not_found", "missing"}:
        return None
    return outcome == "missing"


def cut_before_marker(record: AnswerRecord) -> bool | None:
    """표지 전 잘림인가 — grounded 가 표지를 정하기 전에 max_tokens 에 닿아 아무것도 못 보내고 partial 로 넘어감.

    grounded 를 돌려 done 으로 끝난 기록(done_reason 에 grounded 가 있음)만 센다. 아니면 None.
    """
    if "grounded" not in record.done_reason:
        return None
    return marker_outcome(record) == "cut_before_marker"


def stage_truncated(record: AnswerRecord, stage: str) -> bool | None:
    """그 단계의 생성이 max_tokens 에 닿았나 (done_reason 이 length). 그 단계를 돌려 done 으로 끝난 기록이 아니면 None.

    done 의 truncated(마지막 단계 것)와 달리 grounded 가 잘려 partial 로 넘어간 것, partial 이 빈 채 잘려 고정 거절
    문구로 끝난 것도 잡는다. 생성을 끊었거나 서버가 done_reason 을 주지 않았으면(None) 잘리지 않은 것으로 센다.
    """
    if stage not in record.done_reason:
        return None
    return record.done_reason[stage] == "length"


def marker_mismatch(record: AnswerRecord) -> bool | None:
    """표지는 '찾음'인데 본문이 정보가 없다고 하나. grounded 가 '찾음'으로 답을 낸 기록이 아니면 None.

    본문 어디서든 NO_INFO_PATTERNS 가 걸리면 어긋남으로 센다 — 답 뒤에 곁가지를 '정보 없음'이라 덧붙인 것도 걸리므로
    상한이다. KorQuAD 는 정답 포함과 같이 보면 가른다.
    """
    if record.marker != "found" or final_stage(record) != "grounded":
        return None
    return says_no_info(record.answer)


def has_dropped(record: AnswerRecord) -> bool | None:
    """답에 범위 밖 출처 번호가 있었나 (서버가 뗐다). LLM 이 답을 쓰지 않은 기록이면 None."""
    if not answered(record):
        return None
    return bool(record.dropped)


def matches_expected(record: AnswerRecord) -> bool | None:
    """위키 밖 질문이 기대 단계(EXPECTED_FINAL)로 끝났나. 기대 단계가 없는 문항(KorQuAD)이면 None."""
    if record.item.expected is None:
        return None
    return final_stage(record) in EXPECTED_FINAL[record.item.expected]


def score_band(score: float | None, edges: Sequence[float] = SCORE_BAND_EDGES) -> str:
    """리랭커 1위 점수 구간 이름 ("< 0.83" · "0.83–0.94" · "≥ 0.94"). 점수가 없으면 "없음"."""
    if score is None:
        return "없음"
    if score < edges[0]:
        return f"< {edges[0]:g}"
    for lower, upper in itertools.pairwise(edges):
        if score < upper:
            return f"{lower:g}–{upper:g}"
    return f"≥ {edges[-1]:g}"


def score_bands(edges: Sequence[float] = SCORE_BAND_EDGES) -> tuple[str, ...]:
    """`score_band` 가 내는 구간 이름 전부 — 점수 낮은 순, 끝에 "없음"."""
    return (*(score_band(lower, edges) for lower in (-math.inf, *edges)), score_band(None))


def latency(
    values: Iterable[float | None], percentiles: Sequence[int] = LATENCY_PERCENTILES
) -> Latency:
    """지연 값(ms)의 백분위 (evalkit `percentile` — numpy 기본과 같은 선형 보간). None(그 시간이 없는 문항)은 뺀다."""
    present = [value for value in values if value is not None]
    # evalkit percentile 은 빈 목록에 ValueError — 값이 하나도 없으면 백분위를 비워 둔다
    points = {q: percentile(present, q) for q in percentiles} if present else {}
    return Latency(count=len(present), percentiles=MappingProxyType(points))


def answer_rank(hits: Sequence[Hit], item: KorQuADItem) -> int | None:
    """검색 결과에서 정답 근거 청크(정답 문서이고 정답 문자열을 담음)의 첫 순위 (1부터). 없으면 None."""
    return next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if contains_answer(hit.chunk.title, hit.chunk.text, item)
        ),
        None,
    )


def paired_hits(pairs: Iterable[tuple[bool, bool]]) -> PairedHits:
    """(원문 성공, 패러프레이즈 성공) 짝들을 표로 세고 McNemar 정확 검정을 붙인다. 짝이 없으면 p 값 1."""
    rows = list(pairs)
    counts = Counter(rows)
    test = mcnemar_exact([original for original, _ in rows], [para for _, para in rows])
    return PairedHits(
        both=counts[True, True],
        original_only=test.only_a,
        paraphrase_only=test.only_b,
        neither=counts[False, False],
        mcnemar_p=test.p_value,
    )


def _grounded_sent(record: AnswerRecord) -> bool:
    # 답 조각을 하나라도 보낸 단계는 바뀌지 않는다 (ADR-027) — grounded 로 done 이 났으면 grounded 가 답을 보냈다
    return record.error is None and record.stage_path[-1:] == ("grounded",)
