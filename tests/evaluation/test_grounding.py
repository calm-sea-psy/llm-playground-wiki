import json
from dataclasses import dataclass, replace

import pytest

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.grounding import (
    EXCLUDED,
    FABRICATED,
    UNSUPPORTED,
    VERDICT_LABELS,
    VERDICTS,
    GroundingVerdict,
    cascade_rows,
    cluster_bootstrap_difference,
    judge_rows,
    newcombe_difference,
)
from playground_wiki.evaluation.partial_judge import DOC_HEADING, GENERAL_HEADING

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"


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
    assert tuple(VERDICT_LABELS) == VERDICTS
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


def _record(item: EvalItem, **fields: object) -> AnswerRecord:
    base = AnswerRecord(
        item=item,
        stage_path=("grounded",),
        answer=f"흥선대원군 {ANSWER_MARK} [1].",
        hits=(ContextChunk("1-0", "경복궁", "", f"흥선대원군 {CHUNK_MARK}", 0.97),),
        context_count=1,
        marker="found",
        cited=(1,),
        dropped=(),
        truncated=False,
        timed_out=False,
        error=None,
        top_score=0.97,
        gate_rule=None,
        spec_id="ollama:fake",
        prompt_version="base@v4:x+grounded@v2:y+question@v1:z",
        search_ms=50.0,
        first_delta_ms=900.0,
        total_ms=2000.0,
        wall_ms=2001.0,
        done_reason={"grounded": "stop"},
    )
    return replace(base, **fields)


def _records() -> list[AnswerRecord]:
    palace = EvalItem(
        id="k1",
        source="korquad",
        question=f"경복궁 {QUESTION_MARK}",
        gold_title="경복궁",
        gold_answer="흥선대원군",
    )
    return [_record(palace)]


def _strata_records() -> list[AnswerRecord]:
    base = EvalItem(
        id="m1",
        source="korquad",
        question=QUESTION_MARK,
        gold_title="경복궁",
        gold_answer="흥선대원군",
    )
    return [
        _record(base, top_score=0.85),
        _record(replace(base, id="m2"), top_score=0.88, answer=f"고종 {ANSWER_MARK} [1]."),
        _record(replace(base, id="m3"), top_score=0.90, stage_path=("grounded", "partial")),
        _record(replace(base, id="h1"), top_score=0.99),
        _record(replace(base, id="h2"), top_score=0.97),
    ]


def _verdict(judge_id: str, verdict: str) -> GroundingVerdict:
    return GroundingVerdict.from_json(_row(judge_id=judge_id, verdict=verdict))


@dataclass(frozen=True)
class _Measured:
    """cascade_rows 가 읽는 F9 · 층화 기록 한 벌 (scripts/eval_answer.py 의 Measurement 대신)."""

    f9: tuple[AnswerRecord, ...]
    strata: tuple[AnswerRecord, ...]


def _measurement(
    strata: list[AnswerRecord],
    verdicts: dict[str, GroundingVerdict],
    f9: list[AnswerRecord],
    system: list[str],
) -> _Measured:
    del verdicts, system  # cascade_rows 는 기록만 본다
    return _Measured(f9=tuple(f9), strata=tuple(strata))


def _cascade_measurements() -> tuple[_Measured, _Measured]:
    doc, general = DOC_HEADING, GENERAL_HEADING
    strata = _strata_records()
    base_verdicts = {
        "m1": _verdict("J1", "supported"),
        "m2": _verdict("J2", "misread"),
        "h1": _verdict("J3", "supported"),
        "h2": _verdict("J4", "supported"),
    }
    # m2 · k1 이 다시 잰 쪽에서 못 찾음 → partial. m2 는 문서 칸에, k1 은 일반 지식 칸에만 정답
    strata_answer = f"{doc}\n흥선대원군 {ANSWER_MARK} [1].\n\n{general}\n일반 {ANSWER_MARK}"
    other_strata = [
        replace(r, stage_path=("grounded", "partial"), answer=strata_answer)
        if r.item.id == "m2"
        else r
        for r in strata
    ]
    other_verdicts = {key: base_verdicts[key] for key in ("m1", "h1", "h2")}
    f9 = [_records()[0]]
    f9_answer = f"{doc}\n직접적인 내용은 없습니다. [1]\n\n{general}\n흥선대원군 {ANSWER_MARK}"
    other_f9 = [replace(f9[0], stage_path=("grounded", "partial"), answer=f9_answer)]
    base = _measurement(strata, base_verdicts, f9, ["grounded@v2"])
    other = _measurement(other_strata, other_verdicts, other_f9, ["grounded@v3"])
    return base, other


def test_judge_rows_hide_score_band_gold_and_item_id():
    records = _strata_records()

    rows, key = judge_rows(records, seed=0)

    # grounded 로 끝난 m1 · m2 · h1 · h2 만 (m3 은 못 찾음 → partial)
    assert sorted(row["id"] for row in key) == ["h1", "h2", "m1", "m2"]
    assert [row["judge_id"] for row in rows] == ["J001", "J002", "J003", "J004"]
    assert [row["judge_id"] for row in key] == ["J001", "J002", "J003", "J004"]
    assert set(rows[0]) == {"judge_id", "question", "documents", "answer"}
    assert rows[0]["documents"] == [{"number": 1, "text": f"흥선대원군 {CHUNK_MARK}"}]
    text = json.dumps(rows, ensure_ascii=False)
    for hidden in ("0.85", "0.99", '흥선대원군"', "m1", "h1"):
        assert hidden not in text


def test_cascade_rows_judge_only_the_document_part_and_hide_which_record():
    base, other = _cascade_measurements()

    rows, key = cascade_rows(base, other, seed=0)

    assert [row["judge_id"] for row in rows] == ["C001", "C002", "C003"]
    assert sorted((row["record"], row["side"], row["id"]) for row in key) == [
        ("f9", "base", "k1"),
        ("f9", "tag", "k1"),
        ("strata", "tag", "m2"),
    ]
    answers = {(row["record"], row["side"]): rows[i]["answer"] for i, row in enumerate(key)}
    assert answers[("f9", "tag")] == "직접적인 내용은 없습니다. [1]"
    assert answers[("strata", "tag")] == f"흥선대원군 {ANSWER_MARK} [1]."
    assert set(rows[0]) == {"judge_id", "question", "documents", "answer"}
    text = json.dumps(rows, ensure_ascii=False)
    for hidden in ("###", "일반", "k1", "m2", "0.88", "strata", "tag"):
        assert hidden not in text
