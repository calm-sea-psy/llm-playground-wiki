import pytest
from playground_corpus import Chunk
from playground_llm import ModelSpec, Think
from playground_retrieval.core import Hit

from playground_wiki.generation import AnswerPolicy, Profile, Thresholds, swap_prompt

THRESHOLDS = Thresholds(high=0.9, low=0.5)
GROUNDED = Profile(
    ModelSpec("ollama", "gemma4:12b", {"think": Think.OFF}),
    system=("base@v1", "grounded@v1"),
    user="question@v1",
    timeout_sec=60,
)
PARTIAL = Profile(
    ModelSpec("ollama", "gemma4:12b", {"think": Think.ON}),
    system=("base@v1", "partial@v1"),
    user="question@v1",
    timeout_sec=120,
)
POLICY = AnswerPolicy(THRESHOLDS, {"grounded": GROUNDED, "partial": PARTIAL})


def _hit(score: float, seq: int = 0) -> Hit:
    chunk = Chunk(
        id=f"1-{seq}",
        doc_id="1",
        title="서울",
        section="",
        text="서울\n한양",
        seq=seq,
        token_count=3,
    )
    return Hit(chunk=chunk, score=score)


@pytest.mark.parametrize(
    ("score", "stage"),
    [(0.9, "grounded"), (0.89, "partial"), (0.5, "partial"), (0.49, "refuse"), (None, "refuse")],
)
def test_stage_boundaries(score: float | None, stage: str):
    assert THRESHOLDS.stage(score) == stage


@pytest.mark.parametrize(("high", "low"), [(0.5, 0.9), (1.1, 0.5), (0.9, -0.1)])
def test_thresholds_reject_wrong_order_or_range(high: float, low: float):
    with pytest.raises(ValueError, match="0 <= low <= high <= 1"):
        Thresholds(high=high, low=low)


def test_grounded_uses_grounded_profile_and_drops_hits_below_low():
    decision = POLICY.decide([_hit(0.95, 0), _hit(0.6, 1), _hit(0.3, 2)])

    assert decision.stage == "grounded"
    assert decision.top_score == 0.95
    assert decision.profile is GROUNDED
    assert [hit.score for hit in decision.context] == [0.95, 0.6]


def test_partial_uses_partial_profile_with_think_on():
    decision = POLICY.decide([_hit(0.7, 0), _hit(0.2, 1)])

    assert decision.stage == "partial"
    assert decision.profile is PARTIAL
    assert decision.profile.spec.think.enabled
    assert [hit.score for hit in decision.context] == [0.7]


def test_refuse_calls_no_llm_and_has_no_context():
    decision = POLICY.decide([_hit(0.4, 0), _hit(0.1, 1)])

    assert (decision.stage, decision.profile, decision.context) == ("refuse", None, ())
    assert decision.top_score == 0.4


def test_no_hits_refuses():
    decision = POLICY.decide([])

    assert (decision.stage, decision.top_score, decision.profile) == ("refuse", None, None)


def test_context_is_sorted_by_score_even_if_hits_are_not():
    decision = POLICY.decide([_hit(0.6, 0), _hit(0.95, 1), _hit(0.7, 2)])

    assert [hit.chunk.seq for hit in decision.context] == [1, 2, 0]
    assert decision.top_score == 0.95


@pytest.mark.parametrize("stages", [{"grounded"}, {"grounded", "partial", "refuse"}])
def test_policy_needs_profiles_for_exactly_grounded_and_partial(stages: set[str]):
    with pytest.raises(ValueError, match="refuse calls no LLM"):
        AnswerPolicy(THRESHOLDS, dict.fromkeys(stages, GROUNDED))


def test_profile_rejects_empty_prompts_and_non_positive_timeout():
    with pytest.raises(ValueError, match="prompts must not be empty"):
        Profile(GROUNDED.spec, system=(), user="question@v1", timeout_sec=60)
    with pytest.raises(ValueError, match="prompts must not be empty"):
        Profile(GROUNDED.spec, system=("base@v1",), user="", timeout_sec=60)
    with pytest.raises(ValueError, match="timeout_sec"):
        Profile(GROUNDED.spec, system=("base@v1",), user="question@v1", timeout_sec=0)


def test_swap_prompt_replaces_only_the_stage_block():
    swapped = swap_prompt(POLICY.profiles, "partial@v3")

    assert swapped["partial"].system == ("base@v1", "partial@v3")
    assert swapped["grounded"] == GROUNDED  # 다른 단계는 그대로
    assert swapped["partial"].spec == PARTIAL.spec  # 모델 · 생성 설정도 그대로
    assert POLICY.profiles["partial"].system == ("base@v1", "partial@v1")  # 원본을 고치지 않는다


@pytest.mark.parametrize("ref", ["partial", "partial@", "@v3", ""])
def test_swap_prompt_rejects_a_ref_without_a_version(ref: str):
    with pytest.raises(ValueError, match="단계@버전"):
        swap_prompt(POLICY.profiles, ref)


def test_swap_prompt_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="no profile for stage"):
        swap_prompt(POLICY.profiles, "refuse@v1")
