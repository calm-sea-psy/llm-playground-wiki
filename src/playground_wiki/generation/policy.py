"""3단계 답변 정책: 리랭커 1위 점수로 grounded · partial · refuse 를 고르고 단계별 생성 설정을 붙인다.

이 정책이 라우터 역할을 한다 (ADR-021). refuse 는 LLM 을 부르지 않는다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, get_args

from playground_llm import ModelSpec
from playground_retrieval.core import Hit

Stage = Literal["grounded", "partial", "refuse"]
STAGES: tuple[str, ...] = get_args(Stage)
LLM_STAGES: frozenset[str] = frozenset(STAGES) - {"refuse"}
"""프로필(생성 설정)이 있어야 하는 단계."""


@dataclass(frozen=True)
class Thresholds:
    """리랭커 점수(0~1) 경계. scripts/calibrate_threshold.py 로 정한다."""

    high: float
    """이 이상이면 grounded."""
    low: float
    """이 이상이면 partial, 미만이면 refuse."""

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError(
                f"thresholds must satisfy 0 <= low <= high <= 1, got {self.low=} {self.high=}"
            )

    def stage(self, score: float | None) -> Stage:
        """리랭커 1위 점수의 단계. 검색 결과가 없으면(None) refuse."""
        if score is None or score < self.low:
            return "refuse"
        return "grounded" if score >= self.high else "partial"


@dataclass(frozen=True)
class Profile:
    """단계 하나의 생성 설정. (spec_id, 프롬프트 버전) 이 벤치 · 평가 기록의 키다."""

    spec: ModelSpec
    """모델과 생성 설정 (think · max_tokens · temperature …). think 를 바꾸면 spec_id 가 달라진다."""
    system: tuple[str, ...]
    """system 턴에 이어 붙일 프롬프트 `이름@버전`. 공통 베이스가 먼저, 단계 블록이 뒤. 지시만 — 문서는 넣지 않는다."""
    user: str
    """user 턴 프롬프트 `이름@버전`. 검색한 문서와 질문이 여기에 들어간다."""
    timeout_sec: float
    """답 하나를 기다리는 상한."""

    def __post_init__(self) -> None:
        if not self.system or not self.user:
            raise ValueError("system and user prompts must not be empty")
        if self.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {self.timeout_sec=}")


@dataclass(frozen=True)
class Decision:
    """질의 하나에 대한 정책 판단."""

    stage: Stage
    top_score: float | None
    """리랭커 1위 점수. 검색 결과가 없으면 None."""
    context: tuple[Hit, ...]
    """답변 근거로 넘길 청크 — low 이상인 것만, 점수 내림차순. refuse 면 비어 있다."""
    profile: Profile | None
    """생성 설정. refuse 면 None (LLM 을 부르지 않는다)."""


class AnswerPolicy:
    """검색 결과의 리랭커 점수를 보고 답변 단계 · 근거 · 생성 설정을 고른다.

    예)
        policy = AnswerPolicy(Thresholds(high=0.99, low=0.5), {"grounded": g, "partial": p})
        decision = policy.decide(result.hits)
        if decision.profile is None:
            ...  # refuse — 고정 문구와 시도한 검색어만 돌려준다
    """

    def __init__(self, thresholds: Thresholds, profiles: Mapping[str, Profile]) -> None:
        if set(profiles) != LLM_STAGES:
            raise ValueError(
                f"profiles must be exactly {sorted(LLM_STAGES)} (refuse calls no LLM), "
                f"got {sorted(profiles)}"
            )
        self.thresholds = thresholds
        self.profiles: Mapping[str, Profile] = MappingProxyType(dict(profiles))

    def decide(self, hits: Sequence[Hit]) -> Decision:
        """리랭커 점수(0~1)가 붙은 검색 결과로 단계를 정한다. low 미만 청크는 근거에서 뺀다."""
        top_score = max((hit.score for hit in hits), default=None)
        stage = self.thresholds.stage(top_score)
        relevant = [hit for hit in hits if hit.score >= self.thresholds.low]
        return Decision(
            stage=stage,
            top_score=top_score,
            context=tuple(sorted(relevant, key=lambda hit: hit.score, reverse=True)),
            profile=self.profiles.get(stage),
        )


def swap_prompt(profiles: Mapping[str, Profile], *refs: str) -> dict[str, Profile]:
    """단계 블록 프롬프트를 refs 로 바꾼 프로필 — configs/answer.yaml 은 그대로 두고 다시 잴 때 쓴다.

    ref 의 이름이 단계 이름이다 (`partial@v3` → partial 단계). 그 단계 system 의 마지막 조각(단계 블록)만
    바뀌고 공통 베이스(base@v4)와 refs 에 없는 단계는 그대로다. 여러 단계를 한 번에 바꿀 수 있다
    (`grounded@v4`, `partial@v4` — grounded 가 넘긴 질문을 partial 이 받는 두 프롬프트를 함께 잴 때).
    채택은 측정 뒤 사람이 answer.yaml 에 적는다.

    Args:
      profiles: 단계 → 프로필. 바꾸지 않고 새 dict 를 돌려준다.
      refs: `단계@버전` 꼴의 프롬프트 참조. 단계마다 하나까지.

    Raises:
      ValueError: ref 가 `이름@버전` 꼴이 아니거나, 그 이름의 단계가 profiles 에 없거나, 한 단계에 둘 이상일 때.
    """
    swapped = dict(profiles)
    seen: dict[str, str] = {}
    for ref in refs:
        stage, _, version = ref.partition("@")
        if not stage or not version:
            raise ValueError(f"prompt ref must be '단계@버전', got {ref!r}")
        if stage not in profiles:
            raise ValueError(f"{ref}: no profile for stage {stage!r}, have {sorted(profiles)}")
        if stage in seen:
            raise ValueError(f"stage {stage!r} swapped twice: {seen[stage]} and {ref}")
        seen[stage] = ref
        profile = swapped[stage]
        swapped[stage] = replace(profile, system=(*profile.system[:-1], ref))
    return swapped
