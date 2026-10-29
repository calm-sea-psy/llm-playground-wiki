import pytest

from playground_wiki.evaluation.grounding import (
    EXCLUDED,
    FABRICATED,
    UNSUPPORTED,
    VERDICT_LABELS,
    VERDICTS,
    GroundingVerdict,
    cluster_bootstrap_difference,
    newcombe_difference,
)


def _row(**overrides: object) -> dict[str, object]:
    return {
        "judge_id": "J001",
        "verdict": "supported",
        "claim": "주장",
        "evidence": "[1] 구절",
        "reason": "이유",
        "wrong_citation": False,
        "off_question": False,
        **overrides,
    }


def test_verdict_labels_cover_every_verdict():
    assert set(VERDICT_LABELS) == set(VERDICTS)
    assert FABRICATED < UNSUPPORTED < set(VERDICTS)
    assert not EXCLUDED & UNSUPPORTED


def test_verdict_from_json_defaults_to_the_first_verdict_and_not_reviewed():
    verdict = GroundingVerdict.from_json(_row(verdict="misread"))

    assert verdict.verdict == "misread"
    assert verdict.first_verdict == "misread"
    assert not verdict.reviewed


def test_verdict_from_json_keeps_a_reviewed_change():
    verdict = GroundingVerdict.from_json(
        _row(verdict="supported", first_verdict="not_in_context", reviewed=True)
    )

    assert (verdict.first_verdict, verdict.verdict, verdict.reviewed) == (
        "not_in_context",
        "supported",
        True,
    )


@pytest.mark.parametrize("field", ["verdict", "first_verdict"])
def test_verdict_from_json_rejects_an_unknown_label(field: str):
    with pytest.raises(ValueError, match="verdict must be one of"):
        GroundingVerdict.from_json(_row(**{field: "hallucinated"}))


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        # Newcombe (1998) 표 II — 방법 10
        ((56, 70, 48, 80), (0.2, 0.0524, 0.3339)),
        ((9, 10, 3, 10), (0.6, 0.1705, 0.8090)),
        ((6, 7, 2, 7), (0.5714, 0.0582, 0.8062)),
        ((5, 56, 0, 29), (0.0893, -0.0381, 0.1926)),
        ((0, 10, 0, 20), (0.0, -0.1611, 0.2775)),
        ((10, 10, 0, 20), (1.0, 0.6791, 1.0)),
    ],
)
def test_newcombe_difference_matches_the_published_table(
    counts: tuple[int, int, int, int], expected: tuple[float, float, float]
):
    result = newcombe_difference(*counts)

    assert (result.difference, result.low, result.high) == pytest.approx(expected, abs=5e-5)


def test_cluster_bootstrap_difference_is_reproducible_and_brackets_the_point():
    groups_a = [[True, False], [False], [True, True, False], [False, False]]
    groups_b = [[False], [False, False], [True], [False, False, False]]

    first = cluster_bootstrap_difference(groups_a, groups_b, seed=3)
    second = cluster_bootstrap_difference(groups_a, groups_b, seed=3)

    assert first == second
    assert first.difference == pytest.approx(3 / 8 - 1 / 7)
    assert first.low <= first.difference <= first.high


def test_cluster_bootstrap_difference_of_identical_constant_groups_is_zero():
    groups = [[False, False], [False]]

    result = cluster_bootstrap_difference(groups, groups)

    assert (result.difference, result.low, result.high) == (0.0, 0.0, 0.0)


def test_cluster_bootstrap_difference_rejects_an_empty_group():
    with pytest.raises(ValueError, match="no empty group"):
        cluster_bootstrap_difference([[True], []], [[False]])
