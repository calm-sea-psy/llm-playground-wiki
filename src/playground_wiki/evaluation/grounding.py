"""grounded 답의 지어냄 판정 라벨과, 두 점수 구간의 지어냄 비율 비교 (ADR-024 — high 확정).

라벨은 KorQuAD 정답이 아니라 모델이 받은 문서와 대조해 붙인다 — 같은 사실을 다른 말로 쓰거나 일부만 답한 것은
지어냄이 아니다. 판정은 1위 점수 · 구간 · 정답을 가린 채 한다 (판정 입력은 scripts/eval_answer.py judge-input).
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Self, cast, get_args

from playground_evalkit import percentile, wilson_interval

Verdict = Literal["supported", "extra_unsupported", "misread", "not_in_context", "body_refusal"]
VERDICTS: tuple[str, ...] = get_args(Verdict)
VERDICT_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "supported": "근거 있음",
        "extra_unsupported": "곁가지 근거 없음",
        "misread": "지어냄 · 잘못 읽음",
        "not_in_context": "지어냄 · 문서에 없음",
        "body_refusal": "본문 거절",
    }
)
FABRICATED: frozenset[str] = frozenset({"misread", "not_in_context"})
"""지어냄 — 질문에 답하는 핵심 주장이 문서와 어긋나거나(대상 · 수치 · 시점을 옮김) 문서에 없다 (맞는 일반 지식이어도)."""
UNSUPPORTED: frozenset[str] = FABRICATED | {"extra_unsupported"}
"""곁가지까지 — 핵심은 문서에 있어도 꼭 필요 없는 덧붙임에 문서에 없는 사실이 있으면 넣는다."""
EXCLUDED: frozenset[str] = frozenset({"body_refusal"})
"""분모에서 빼는 라벨 — 표지는 '찾음'인데 본문이 문서에 답이 없다고 했다 (답을 내지 않았다)."""


@dataclass(frozen=True)
class GroundingVerdict:
    """답 하나의 판정 (f9-strata-verdicts JSONL 한 줄)."""

    judge_id: str
    """판정 입력의 가린 번호. 문항 id 와는 판정 열쇠 파일로 잇는다."""
    verdict: Verdict
    """최종 라벨 (메인 재검토 뒤)."""
    claim: str
    """답에서 뽑은, 질문에 답하는 핵심 주장."""
    evidence: str
    """대조한 문서 구절과 출처 번호. 문서에 없으면 빈 문자열."""
    reason: str
    """라벨을 고른 이유 한두 문장."""
    wrong_citation: bool
    """출처 번호가 그 사실이 없는 문서를 짚었다 (사실은 다른 문서에 있을 수 있다)."""
    off_question: bool
    """질문과 다른 것에 답했다 (그 답 자체는 문서에 있을 수 있다)."""
    first_verdict: Verdict
    """판정자가 처음 붙인 라벨. 메인 재검토로 바뀌었으면 verdict 와 다르다."""
    reviewed: bool
    """메인 세션이 다시 봤다."""

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """판정 JSONL 한 줄에서. first_verdict 가 없으면 verdict, reviewed 가 없으면 False.

        Raises:
          ValueError: 라벨이 Verdict 에 없을 때.
        """
        verdict = _verdict(row["verdict"])
        return cls(
            judge_id=str(row["judge_id"]),
            verdict=verdict,
            claim=str(row["claim"]),
            evidence=str(row["evidence"]),
            reason=str(row["reason"]),
            wrong_citation=bool(row["wrong_citation"]),
            off_question=bool(row["off_question"]),
            first_verdict=_verdict(row.get("first_verdict", verdict)),
            reviewed=bool(row.get("reviewed", False)),
        )


@dataclass(frozen=True)
class RateDifference:
    """두 독립 표본의 비율 차이 (a − b) 와 신뢰구간."""

    difference: float
    """비율(a) − 비율(b). 양수면 a 가 높다."""
    low: float
    """구간 하한."""
    high: float
    """구간 상한."""


def newcombe_difference(
    count_a: int, total_a: int, count_b: int, total_b: int, *, confidence: float = 0.95
) -> RateDifference:
    """독립인 두 비율 count_a/total_a − count_b/total_b 와 Newcombe (1998) 방법 10 구간 — 각 비율의 Wilson 구간을 합친다.

    두 구간의 문항이 서로 달라 짝 검정(McNemar)을 쓸 수 없을 때. 0건 · 전체 성공에서도 구간이 [-1, 1] 안에 있다.
    """
    rate_a, rate_b = count_a / total_a, count_b / total_b
    low_a, high_a = wilson_interval(count_a, total_a, confidence=confidence)
    low_b, high_b = wilson_interval(count_b, total_b, confidence=confidence)
    difference = rate_a - rate_b
    return RateDifference(
        difference=difference,
        low=difference - math.hypot(rate_a - low_a, high_b - rate_b),
        high=difference + math.hypot(high_a - rate_a, rate_b - low_b),
    )


def cluster_bootstrap_difference(
    groups_a: Sequence[Sequence[bool]],
    groups_b: Sequence[Sequence[bool]],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateDifference:
    """묶음(문단)을 복원 추출하는 bootstrap 으로 비율 차이 a − b 의 백분위 구간을 돌려준다.

    같은 문단의 질문은 문맥을 나눠 써서 서로 독립이 아니다 — 문항이 아니라 문단을 뽑아 구간이 얼마나 넓어지는지 본다.

    Args:
      groups_a: a 쪽 묶음마다 문항의 성공(지어냄) 여부. 빈 묶음이 있으면 안 된다.
      groups_b: b 쪽, 같은 꼴.
      resamples: 다시 뽑는 횟수.
      confidence: 구간의 신뢰 수준 (0~1).
      seed: 같은 seed 면 같은 구간.
    """
    if not groups_a or not groups_b or not all(groups_a) or not all(groups_b):
        raise ValueError("groups must be non-empty and contain no empty group")
    rng = random.Random(seed)
    differences = [
        _resampled_rate(groups_a, rng) - _resampled_rate(groups_b, rng) for _ in range(resamples)
    ]
    tail = (1 - confidence) / 2 * 100
    return RateDifference(
        difference=_rate(groups_a) - _rate(groups_b),
        low=percentile(differences, tail),
        high=percentile(differences, 100 - tail),
    )


def _resampled_rate(groups: Sequence[Sequence[bool]], rng: random.Random) -> float:
    return _rate(rng.choices(groups, k=len(groups)))


def _rate(groups: Sequence[Sequence[bool]]) -> float:
    return sum(sum(group) for group in groups) / sum(len(group) for group in groups)


def _verdict(value: object) -> Verdict:
    if value not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {value!r}")
    return cast("Verdict", value)
