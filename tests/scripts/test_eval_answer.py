"""scripts/eval_answer.py 의 인자 · 리포트를 합성 입력으로 — 데이터 · 모델 없이 돈다.

표본 · 층화 · 판정 입력은 src 로 옮겨 tests/evaluation (test_f9_sample · test_strata · test_grounding) 에서 본다.
"""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from conftest import load_script

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.files import write_jsonl
from playground_wiki.evaluation.grounding import GroundingVerdict
from playground_wiki.evaluation.partial_judge import (
    DOC_HEADING,
    GENERAL_HEADING,
    NO_DIRECT,
    UNKNOWN,
)

eval_answer = load_script("eval_answer")

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"


def test_title_fixes_point_to_existing_style_titles():
    assert dict(eval_answer.KORQUAD_TITLE_FIXES) == {
        "아가씨": "아가씨 (영화)",
        "조현아": "조승연 (기업인)",
    }


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
    personal = EvalItem(
        id="o1", source="offwiki", question=QUESTION_MARK, expected="refuse", category="personal"
    )
    no_llm = {"marker": None, "spec_id": None, "first_delta_ms": None, "done_reason": {}}
    return [
        _record(palace),
        # 표지 전 잘림 → partial (done.truncated 는 partial 것이라 False)
        _record(
            replace(palace, id="k2"),
            stage_path=("grounded", "partial"),
            marker="missing",
            top_score=0.9,
            done_reason={"grounded": "length", "partial": "stop"},
        ),
        _record(replace(palace, id="k3"), error="generation", total_ms=None, **no_llm),
        # 답을 보내던 중 한도에 닿음
        _record(replace(palace, id="k4"), truncated=True, done_reason={"grounded": "length"}),
        _record(personal, stage_path=("gate",), hits=(), top_score=None, **no_llm),
        # partial 이 빈 채 잘려 고정 거절 문구
        _record(
            replace(personal, id="o2"),
            stage_path=("partial",),
            top_score=0.3,
            cited=(),
            **{**no_llm, "done_reason": {"partial": "length"}},
        ),
    ]


RUNS = [
    {
        "started_at": "2026-10-25T00:00:00+00:00",
        "limit": None,
        "settings": {
            "collection": "kowiki_kure-v1",
            "model": "kure-v1",
            "candidates": 50,
            "rerank_candidates": 20,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "top_k": 5,
            "thresholds": {"high": 0.824914, "low": 0.062973},
            "gate": True,
            "dump_date": "2026년 10월 1일",
            "profiles": {},
            "inputs": {"sample": "data/eval/f9-sample-v1.jsonl"},
        },
    }
]


def test_report_has_every_section_and_no_raw_text():
    sensitivity = (
        [
            {"id": "a", "original": {"answer_rank": 1}, "paraphrase": {"answer_rank": None}},
            {"id": "b", "original": {"answer_rank": 3}, "paraphrase": {"answer_rank": 2}},
        ],
        {**RUNS[0]["settings"], "created_at": "2026-10-25T00:00:00+00:00"},
    )

    text = eval_answer.render_report(_records(), RUNS, 6, sensitivity, today=date(2026, 10, 25))

    for heading in (
        "# F9 답변 평가 — 자동 지표 (2026-10-25)",
        "## 단계 경로",
        "## KorQuAD — 근거 답변",
        "## 리랭커 1위 점수 구간",
        "## 표지 · 출처 번호 · 잘림",
        "## 지연",
        "## 위키 밖 질문",
        "## 검색의 표현 민감도",
        "## 다음 일",
    ):
        assert heading in text
    assert "| hit@5 | 100.0% | 50.0% | +50.0 |" in text
    assert "뺌" not in text  # 뜻이 달라진 문항이 표본에 없으면 줄을 더하지 않는다
    assert eval_answer.PARAPHRASE_REVIEW in text
    assert "generation 1" in text
    for raw in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK):
        assert raw not in text


def test_report_overview_names_the_record_and_the_grounded_prompt():
    profiles = {"grounded": {"system": ["base@v4", "grounded@v4"]}}
    runs = [{**RUNS[0], "settings": {**RUNS[0]["settings"], "profiles": profiles}}]

    text = eval_answer.render_report(
        _records(),
        runs,
        6,
        None,
        today=date(2026, 10, 31),
        answers_path=Path("data/eval/f9-answers-v1-safety-v4-1.jsonl"),
    )

    assert "→ grounded(`base@v4 + grounded@v4`) 첫 줄 표지" in text
    assert "로컬 기록(data/eval/f9-answers-v1-safety-v4-1.jsonl)" in text


def test_report_overview_without_profiles_names_no_prompt():
    text = eval_answer.render_report(_records(), RUNS, 6, None, today=date(2026, 10, 25))

    assert "→ grounded 첫 줄 표지" in text
    assert "로컬 기록(data/eval/f9-answers-v1.jsonl)" in text


def test_report_accepts_tag():
    assert "report" in eval_answer.TAG_STEPS


def test_report_separates_truncation_by_stage_and_the_fixed_refusal():
    text = eval_answer.render_report(_records(), RUNS, 6, None, today=date(2026, 10, 25))

    assert "| 표지 누락 | 0/2 (0.0%) |" in text  # 표지 전 잘림(k2)은 누락 분모에서 빠진다
    assert "| 표지 전 잘림 | 1/3 (33.3%) |" in text
    assert "| grounded 잘림 (max_tokens 도달) | 2/3 (66.7%) |" in text  # k2 · k4
    assert "| partial 잘림 (max_tokens 도달) | 1/2 (50.0%) |" in text  # 고정 거절로 끝난 o2
    assert "| 고정 거절 문구로 끝남 (partial 빈 답) | 1/2 (50.0%) |" in text
    # LLM 이 쓴 답 = k1 · k2 · k4 — 관문 o1 · 고정 거절 o2 · 실패 k3 은 빠진다
    assert "| 범위 밖 출처 번호를 뗀 답 | 0/3 (0.0%) |" in text
    assert "grounded 첫 줄: 찾음 2 · 못 찾음 0 · 표지 누락 0 · 표지 전 잘림 1 · 빈 답 0" in text
    assert "| partial → 고정 거절 |" in text
    assert "| refuse/personal | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 2/2 (100.0%) |" in text
    assert "예전 기록" not in text


def _partial_expected(qid: str, **fields: object) -> AnswerRecord:
    item = EvalItem(
        id=qid,
        source="offwiki",
        question=QUESTION_MARK,
        expected="partial",
        category="missing_fact",
    )
    return _record(item, **fields)


def test_report_counts_the_review_of_partial_expected_answers_that_ended_grounded():
    reviewed = [
        _partial_expected(qid, prompt_version=eval_answer.OFFWIKI_REVIEW_PROMPT)
        for qid in sorted(eval_answer.OFFWIKI_REVIEW)
    ]
    records = [*_records(), *reviewed, _partial_expected("ow-999")]

    text = eval_answer.render_report(records, RUNS, 10, None, today=date(2026, 10, 25))

    assert "4개 중 3개 — 근거 있음 2, 지어냄 · 잘못 읽음 1. 덧표시 질문을 비껴감 1" in text
    assert eval_answer.OFFWIKI_REVIEW_NOTE in text


def test_report_does_not_reuse_the_review_for_answers_from_another_prompt():
    (qid, *_) = sorted(eval_answer.OFFWIKI_REVIEW)
    records = [*_records(), _partial_expected(qid, prompt_version="base@v4:x+grounded@v3:y")]

    text = eval_answer.render_report(records, RUNS, 7, None, today=date(2026, 10, 25))

    assert "그 답의 확인: 아직 (1개)" in text
    assert eval_answer.OFFWIKI_REVIEW_NOTE not in text
    # partial 기대가 grounded 로 끝난 답이 없으면 줄을 더하지 않는다
    assert "그 답의 확인" not in eval_answer.render_report(
        _records(), RUNS, 6, None, today=date(2026, 10, 25)
    )


def test_report_notes_old_records_without_done_reason():
    records = [replace(record, done_reason={}) for record in _records()]

    text = eval_answer.render_report(records, RUNS, 6, None, today=date(2026, 10, 25))

    # LLM 을 불러 done 으로 끝난 k1 · k2 · k4 · o2 (실패 k3 은 원래 없고, 관문 o1 은 LLM 을 안 불렀다)
    assert "done_reason 이 없는 예전 기록 4개" in text
    assert "| 표지 전 잘림 | 0/0 |" in text


def test_sensitivity_adds_a_row_without_paraphrases_whose_meaning_shifted():
    (shifted, *_) = sorted(eval_answer.PARAPHRASE_SHIFTED)
    rows = [
        {"id": "a", "original": {"answer_rank": 1}, "paraphrase": {"answer_rank": None}},
        {"id": "b", "original": {"answer_rank": 3}, "paraphrase": {"answer_rank": 2}},
        {"id": shifted, "original": {"answer_rank": None}, "paraphrase": {"answer_rank": 1}},
    ]

    text = eval_answer.render_report(
        _records(), RUNS, 6, (rows, RUNS[0]["settings"]), today=date(2026, 10, 25)
    )

    assert "| hit@5 | 66.7% | 66.7% | +0.0 |" in text  # 전부로 잰 값이 먼저
    assert "| hit@5 · 뜻 달라짐 1개 뺌 | 100.0% | 50.0% | +50.0 |" in text


def test_report_marks_sensitivity_as_not_run_and_mixed_settings():
    changed = {**RUNS[0], "settings": {**RUNS[0]["settings"], "top_k": 10}}

    text = eval_answer.render_report(
        _records(), [*RUNS, changed], 5, None, today=date(2026, 10, 25)
    )

    assert "아직 돌리지 않았다" in text
    assert "설정이 바뀌었다" in text


def _pool_row(qid: str, paragraph: str, score: float | None) -> dict:
    return {
        "id": qid,
        "paragraph": paragraph,
        "top_score": score,
        "in_corpus": True,
        "answer_in_document": True,
        "answer_rank": 1,
    }


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


def _verdict_row(judge_id: str, verdict: str) -> dict:
    return {
        "judge_id": judge_id,
        "verdict": verdict,
        "claim": f"주장 {ANSWER_MARK}",
        "evidence": f"[1] {CHUNK_MARK}",
        "reason": "이유",
        "wrong_citation": False,
        "off_question": False,
    }


def _verdict(judge_id: str, verdict: str) -> GroundingVerdict:
    return GroundingVerdict.from_json(_verdict_row(judge_id, verdict))


def test_strata_report_compares_fabrication_by_band_without_raw_text():
    records = _strata_records()
    verdicts = {
        "m1": _verdict("J1", "supported"),
        "m2": _verdict("J2", "misread"),
        "h1": _verdict("J3", "supported"),
        "h2": replace(_verdict("J4", "extra_unsupported"), first_verdict="misread", reviewed=True),
    }
    pool_rows = [
        _pool_row(record.item.id, f"문서{index}#0", 0.9) for index, record in enumerate(records)
    ]
    runs = [
        {
            **RUNS[0],
            "settings": {
                **RUNS[0]["settings"],
                "profiles": {"grounded": {"spec_id": "ollama:fake"}},
            },
        }
    ]

    text = eval_answer.render_strata_report(
        records, runs, pool_rows, verdicts, today=date(2026, 10, 29)
    )

    for heading in (
        "# high 확정",
        "## 점수 풀",
        "## 층화 표본 — 단계 경로",
        "## 지어냄 판정",
        "## 두 구간 비교",
        "## 판정 방법 · 검수",
        "## 결론",
    ):
        assert heading in text
    assert "| 0.83–0.94 | 2 | 1 | 0 | 1 | 0 | 0 | 1/2 (50.0%," in text
    assert "| ≥ 0.94 | 2 | 1 | 1 | 0 | 0 | 0 | 0/2 (0.0%," in text
    assert "| 지어냄 | 1/2 (50.0%) | 0/2 (0.0%) | +50.0 |" in text
    assert "| 곁가지 포함 | 1/2 (50.0%) | 1/2 (50.0%) | +0.0 |" in text
    assert "다시 본 것 1, 바뀐 판정 1 (지어냄 · 잘못 읽음 → 곁가지 근거 없음 1)" in text
    # 점수 풀 0.9 로 넣었으므로 다시 검색한 점수가 ≥ 0.94 인 h1 · h2 가 구간을 옮겼다
    assert "다른 구간에 든 문항 2개" in text
    for raw in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK):
        assert raw not in text


def test_strata_report_crosses_fabrication_with_gold_in_the_top_five():
    records = _strata_records()
    verdicts = {
        "m1": _verdict("J1", "supported"),
        "m2": _verdict("J2", "misread"),
        "h1": _verdict("J3", "supported"),
        "h2": _verdict("J4", "misread"),
    }
    pool_rows = [_pool_row(record.item.id, f"문서{i}#0", 0.9) for i, record in enumerate(records)]
    pool_rows[1]["answer_rank"] = None  # m2: 점수 풀 검색 상위 5 에 정답 근거가 없었다
    pool_rows.append(_pool_row("x1", "문서9#0", 0.826))  # high 이상 · 0.83 미만 — 표본 구간 밖
    runs = [
        {
            **RUNS[0],
            "settings": {**RUNS[0]["settings"], "profiles": {"grounded": {"spec_id": "fake"}}},
        }
    ]

    text = eval_answer.render_strata_report(
        records, runs, pool_rows, verdicts, today=date(2026, 10, 29)
    )

    assert "## 지어냄 × 검색 상위 5 정답 근거" in text
    assert "| 0.83–0.94 | 없음 | 1 | 1/1 (100.0%," in text
    assert "| 0.83–0.94 | 있음 | 1 | 0/1 (0.0%," in text
    assert "| ≥ 0.94 | 있음 | 2 | 1/2 (50.0%," in text
    assert "0.83–0.94 구간 지어냄 1 중 1문항이" in text
    assert "[0.824914, 0.83) 은 운영에서는 grounded 로 가지만" in text
    assert "점수 풀 1문항" in text
    assert "정답 미포함 · 지어냄 아님 0/4 에는" in text  # m2 만 정답 미포함이고 지어냄이다


def test_strata_report_needs_a_verdict_for_every_grounded_answer():
    records = _strata_records()
    pool_rows = [_pool_row(record.item.id, "문서#0", 0.9) for record in records]

    with pytest.raises(ValueError, match="have no verdict"):
        eval_answer.render_strata_report(records, RUNS, pool_rows, {}, today=date(2026, 10, 29))


def _runs_with(system: list[str]) -> tuple[dict, ...]:
    settings = {**RUNS[0]["settings"], "profiles": {"grounded": {"system": system}}}
    return ({**RUNS[0], "settings": settings},)


def _measurement(
    strata: list[AnswerRecord],
    verdicts: dict[str, GroundingVerdict],
    f9: list[AnswerRecord],
    system: list[str],
) -> object:
    return eval_answer.Measurement(
        f9=tuple(f9),
        strata=tuple(strata),
        verdicts=verdicts,
        f9_runs=_runs_with(system),
        strata_runs=_runs_with(system),
    )


def _offwiki_records(found: set[str]) -> list[AnswerRecord]:
    item = EvalItem(
        id="ow-1", source="offwiki", question=QUESTION_MARK, expected="partial", category="trap"
    )
    records = []
    for qid in ("ow-1", "ow-2"):
        path = ("grounded",) if qid in found else ("grounded", "partial")
        records.append(_record(replace(item, id=qid), stage_path=path))
    return records


def test_compare_report_pairs_the_same_items_without_raw_text():
    strata = _strata_records()  # m3 만 grounded → partial
    base_verdicts = {
        "m1": _verdict("J1", "supported"),
        "m2": _verdict("J2", "misread"),
        "h1": _verdict("J3", "supported"),
        "h2": _verdict("J4", "misread"),
    }
    # 다시 잰 쪽: m2 가 못 찾음으로 partial 에 넘어가고, h2 는 근거 있음
    other_strata = [
        replace(record, stage_path=("grounded", "partial")) if record.item.id == "m2" else record
        for record in strata
    ]
    other_verdicts = {
        "m1": _verdict("K1", "supported"),
        "h1": _verdict("K2", "supported"),
        "h2": _verdict("K3", "supported"),
    }
    korquad = _strata_records()[:2]
    base = _measurement(
        strata, base_verdicts, korquad + _offwiki_records({"ow-1"}), ["base@v4", "grounded@v2"]
    )
    other = _measurement(
        other_strata, other_verdicts, korquad + _offwiki_records(set()), ["base@v4", "grounded@v3"]
    )

    text = eval_answer.render_compare_report(base, other, tag="v3", today=date(2026, 10, 29))

    for heading in (
        "# grounded 프롬프트 비교",
        "## 층화 표본",
        "## F9 KorQuAD",
        "## F9 위키 밖 질문",
        "## 판정 방법 · 검수 (`v3`)",
        "## 확인한 것",
        "## 결론",
    ):
        assert heading in text
    assert (
        '`profiles.grounded.system`: ["base@v4", "grounded@v2"] → ["base@v4", "grounded@v3"]'
        in text
    )
    assert "| 0.83–0.94 | 기준 | 3 | 2 | 1 | 0 | 1/2 (50.0%," in text
    assert "| 0.83–0.94 | v3 | 3 | 1 | 2 | 0 | 0/1 (0.0%," in text
    assert "| 지어냄 답을 냄 | 2/5 (40.0%) | 0/5 (0.0%) | 2 | 0 |" in text
    assert "| grounded 로 끝남 | 4/5 (80.0%) | 3/5 (60.0%) | 1 | 0 |" in text
    assert "| partial | 2 | 1 | 0 | 1/2 (50.0%) | 2/2 (100.0%) |" in text
    assert "| ow-1 | partial/trap | grounded | grounded → partial |" in text
    assert "ow-2" not in text  # 두 기록 모두 grounded 로 끝나지 않았다
    for raw in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK):
        assert raw not in text


def _cascade_measurements() -> tuple[object, object]:
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


def test_compare_report_pairs_verdicts_of_answers_moved_to_partial():
    base, other = _cascade_measurements()
    cascade = {
        ("f9", "base", "k1"): _verdict("C1", "supported"),
        ("f9", "tag", "k1"): _verdict("C2", "body_refusal"),
        ("strata", "tag", "m2"): replace(
            _verdict("C3", "supported"), first_verdict="misread", reviewed=True
        ),
    }

    text = eval_answer.render_compare_report(
        base, other, tag="v3", today=date(2026, 10, 30), cascade=cascade
    )
    without = eval_answer.render_compare_report(base, other, tag="v3", today=date(2026, 10, 30))

    assert "## grounded 에서 partial 로 넘어간 답 — 문서 칸 판정" in text
    assert "## grounded 에서 partial 로 넘어간 답" not in without
    assert "| 층화 표본 | 지어냄 · 잘못 읽음 | 근거 있음 | 1 |" in text
    assert "| F9 KorQuAD | 근거 있음 | 본문 거절 | 1 |" in text
    assert "| 층화 표본 | 1 | 0/1 (0.0%) | 1 | 0 | 0 | 0 |" in text
    assert "| F9 KorQuAD | 1 | 1/1 (100.0%) | 0 | 1 | 0 | 0 |" in text
    assert "판정 3 중 메인 세션이 다시 본 것 1, 바뀐 판정 1" in text
    for raw in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK):
        assert raw not in text


def test_compare_report_needs_the_same_items_in_both_records():
    strata = _strata_records()
    verdicts = {
        "m1": _verdict("J1", "supported"),
        "m2": _verdict("J2", "supported"),
        "h1": _verdict("J3", "supported"),
        "h2": _verdict("J4", "supported"),
    }
    base = _measurement(strata, verdicts, [], ["grounded@v2"])
    other = _measurement(strata[:-1], verdicts, [], ["grounded@v3"])

    with pytest.raises(ValueError, match="cover different items"):
        eval_answer.render_compare_report(base, other, tag="v3", today=date(2026, 10, 29))


def test_ids_option_takes_a_comma_list_only_for_run_and_tally():
    argv = ["run", "--ids", "ow-049, ow-064", "--tag", "noise-v2-1"]
    assert eval_answer._parse_args(argv).ids == ("ow-049", "ow-064")

    for bad in (["report", "--ids", "ow-049"], ["run", "--ids", "ow-049,ow-049"]):
        with pytest.raises(SystemExit):
            eval_answer._parse_args(bad)


def _offwiki_round(found: set[str], *, marker_missing: str = "") -> list[AnswerRecord]:
    item = EvalItem(
        id="ow-049", source="offwiki", question=QUESTION_MARK, expected="partial", category="x"
    )
    records = []
    for qid in ("ow-049", "ow-064"):
        if qid == marker_missing:
            fields = {"marker": "missing"}
        elif qid in found:
            fields = {}
        else:
            fields = {"stage_path": ("grounded", "partial"), "marker": "not_found"}
        records.append(_record(replace(item, id=qid), **fields))
    return records


def test_tally_counts_paths_and_markers_per_round_and_per_item():
    rounds = {
        "noise-v2-1": _offwiki_round({"ow-049", "ow-064"}),
        "noise-v2-2": _offwiki_round({"ow-049"}),
        "noise-v3-1": _offwiki_round(set()),
        "noise-v3-2": _offwiki_round(set(), marker_missing="ow-049"),
    }
    prompts = {tag: "grounded@v2" if "v2" in tag else "grounded@v3" for tag in rounds}

    text = eval_answer.render_tally(rounds, prompts)

    assert "| noise-v2-1 | `grounded@v2` | 2 | 2 | 0 |  | 2 | 0 |  |" in text
    assert "| noise-v3-2 | `grounded@v3` | 2 | 1 | 1 |  | 0 | 1 | 표지 누락 1 |" in text
    assert (
        "| noise-v2 | ow-064 | 2 | 1 | 1 | grounded · 찾음 | grounded → partial · 못 찾음 |" in text
    )
    assert (
        "| noise-v3 | ow-049 | 2 | 1 | 1 | grounded → partial · 못 찾음 | grounded · 표지 누락 |"
        in text
    )
    assert QUESTION_MARK not in text
    assert ANSWER_MARK not in text


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--closed-book"],  # 꼬리표 없이는 기준 기록을 덮어쓴다
        ["run", "--closed-book", "--tag", "cb", "--prompt", "grounded@v2"],
        ["run", "--longtail", "--strata"],
        ["report", "--closed-book"],
        ["run", "--rag-tag", "x"],
        ["closed-book-report"],  # closed-book 기록의 꼬리표가 있어야 한다
        ["report", "--report-date", "2026-10-31"],
    ],
)
def test_closed_book_options_are_checked(argv: list[str]):
    with pytest.raises(SystemExit):
        eval_answer._parse_args(argv)


def test_closed_book_options_parse_for_run_and_the_report():
    argv = ["run", "--closed-book", "--longtail", "--tag", "cb1", "--ids", "a"]
    args = eval_answer._parse_args(argv)
    report = ["closed-book-report", "--tag", "cb1", "--rag-tag", "grounded-v3"]
    report_args = eval_answer._parse_args([*report, "--report-date", "2026-11-01"])

    assert (args.closed_book, args.longtail, args.tag) == (True, True, "cb1")
    assert eval_answer._run_paths(strata=False, longtail=True) == (
        eval_answer.LONGTAIL_SET,
        eval_answer.LONGTAIL_ANSWERS_PATH,
    )
    assert (report_args.rag_tag, report_args.report_date) == ("grounded-v3", date(2026, 11, 1))


def _closed_record(item: EvalItem, answer: str) -> AnswerRecord:
    return _record(
        item,
        stage_path=("closed_book",),
        answer=answer,
        hits=(),
        context_count=0,
        marker=None,
        cited=(),
        top_score=None,
        prompt_version="closed_book@v1:abc",
        done_reason={"closed_book": "stop"},
    )


def _partial_answer(general: str) -> str:
    return f"{DOC_HEADING}\n{NO_DIRECT}. [1]\n\n{GENERAL_HEADING}\n{general}"


def _closed_book_sides() -> tuple[object, object]:
    titled = EvalItem(
        id="k1",
        source="korquad",
        question=f"경복궁 {QUESTION_MARK}",
        gold_title="경복궁",
        gold_answer="흥선대원군",
    )
    deictic, untitled = replace(titled, id="k2"), replace(titled, id="k3")
    personal = EvalItem(
        id="o1", source="offwiki", question=QUESTION_MARK, expected="refuse", category="personal"
    )
    after_dump = replace(personal, id="o2", expected="partial", category="after_dump")
    no_llm = {"marker": None, "spec_id": None, "first_delta_ms": None, "done_reason": {}}
    other = ContextChunk("2-0", "창덕궁", "", f"태종 {CHUNK_MARK}", 0.9)
    rag = (
        _record(titled),  # 정답 포함
        # 못 찾음 → partial 두 칸 정해진 문구 = 모름. 검색 상위에는 정답 근거가 있었다
        _record(
            deictic,
            stage_path=("grounded", "partial"),
            marker="not_found",
            answer=_partial_answer(f"{UNKNOWN}."),
        ),
        _record(untitled, answer=f"고종 {ANSWER_MARK} [1].", hits=(other,), top_score=0.9),
        _record(personal, stage_path=("gate",), hits=(), top_score=None, **no_llm),
        _record(after_dump, stage_path=("partial",), answer=_partial_answer(f"추측 {ANSWER_MARK}")),
    )
    closed = (
        _closed_record(titled, "모름"),
        _closed_record(deictic, f"흥선대원군 {ANSWER_MARK}"),
        _closed_record(untitled, "흥선대원군입니다."),
        _closed_record(personal, "모름."),
        _closed_record(after_dump, f"추측 {ANSWER_MARK}"),
    )
    rag_runs = _runs_with(["base@v4", "grounded@v3"])
    rag_runs[0]["settings"]["profiles"]["partial"] = {"system": ["base@v4", "partial@v2"]}
    closed_runs = (
        {
            "started_at": "2026-11-01T00:00:00+00:00",
            "limit": None,
            "ids": None,
            "settings": {"spec_id": "ollama/fake", "prompt_version": "closed_book@v1:abc"},
        },
    )
    f9 = eval_answer.SidePair(rag=rag, closed=closed, rag_runs=rag_runs, closed_runs=closed_runs)
    director = EvalItem(
        id="longtail:Q2-P57",
        source="longtail",
        question=QUESTION_MARK,
        gold_title="가행복",
        gold_answer="마르셀 샤를 레르비에",
        gold_aliases=("마르셀 레르비에",),
        category="감독",
    )
    longtail = eval_answer.SidePair(
        # 일반 지식 칸에만 다른 표기로 정답
        rag=(
            _record(director, stage_path=("partial",), answer=_partial_answer("마르셀 레르비에")),
        ),
        closed=(_closed_record(director, "마르셀 레르비에입니다."),),
        rag_runs=rag_runs,
        closed_runs=closed_runs,
    )
    return f9, longtail


KINDS = {"k1": "titled", "k2": "deictic", "k3": "untitled"}


def test_closed_book_report_pairs_rag_and_closed_book_without_raw_text():
    f9, longtail = _closed_book_sides()

    text = eval_answer.render_closed_book_report(
        f9, longtail, KINDS, tag="cb1", rag_tag="grounded-v3", today=date(2026, 11, 1)
    )

    for heading in (
        "# RAG 없이 기준선 — closed-book vs RAG (2026-11-01)",
        "## 정답 포함 · 모름 · 답했는데 틀림",
        "## 같은 문항 짝 — 정답 포함",
        "## 결과 교차",
        "## RAG 가 망친 문항",
        "## partial 일반 지식 칸",
        "## 위키 밖 질문",
        "## closed-book 답의 꼴",
        "## 문단 없이 뜻이 안 통하는 KorQuAD 질문",
        "## 결론",
    ):
        assert heading in text
    assert "`ollama/fake`" in text
    assert "`closed_book@v1:abc`" in text
    assert "기록 이름 끝 -grounded-v3" in text
    rag_row = "| KorQuAD 전부 | RAG | 3 | 1/3 (33.3%) | 1/3 (33.3%) | 1/3 (33.3%) | 0/3 (0.0%) | 1/2 (50.0%,"
    assert rag_row in text
    closed_row = "| KorQuAD 전부 | closed-book | 3 | 2/3 (66.7%) | 1/3 (33.3%) | 0/3 (0.0%) | 0/3 (0.0%) | 0/2"
    assert closed_row in text
    assert "| KorQuAD 전부 | 3 | 1/3 (33.3%) | 2/3 (66.7%) | -33.3 | 0 | 1 | 2 | 0 | 1 |" in text
    assert "| 롱테일 | 1 | 1/1 (100.0%) | 1/1 (100.0%) | +0.0 | 1 | 0 | 0 | 0 | 1 |" in text
    assert "| KorQuAD | 정답 포함 | 0 | 1 | 0 | 0 |" in text  # k1: RAG 정답 · closed-book 모름
    assert "| k2 | KorQuAD · 지시어 질문 | grounded → partial | 모름 | 있음 | ≥ 0.94 |" in text
    assert "| k3 | KorQuAD · 제목 없음 | grounded | 답했는데 틀림 | 없음 | 0.83–0.94 |" in text
    assert "2문항 중 검색 상위에 정답 근거 청크가 없던 것 1" in text
    assert "| KorQuAD | 1 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |" in text
    assert "| 롱테일 | 1 | 0 | 1 | 1/1 (100.0%) | 1/1 (100.0%) | 0 | 0 |" in text
    assert "| partial | 1 | 0/1 (0.0%) | 0 | 0/1 (0.0%) | 1/1 (100.0%) |" in text
    assert "| refuse | 1 | 1/1 (100.0%) | 0 | 1/1 (100.0%) | 1/1 (100.0%) |" in text
    assert "| F9 | 5 | 2 | 0 | 0 | 0 | 0 |" in text
    assert "제목 있음 1 · 제목 없음 1 · 지시어 질문 1" in text
    for raw in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK):
        assert raw not in text


def test_closed_book_report_waits_for_longtail_and_needs_the_same_items():
    f9, _ = _closed_book_sides()

    text = eval_answer.render_closed_book_report(
        f9, None, KINDS, tag="cb1", rag_tag=None, today=date(2026, 11, 1)
    )
    short = replace(f9, closed=f9.closed[:-1])

    assert "롱테일 아직 (기록 없음)" in text
    assert "| 롱테일 |" not in text
    assert "기록 이름 끝 없음 (꼬리표 없는 기록)" in text
    with pytest.raises(ValueError, match="different items"):
        eval_answer.render_closed_book_report(
            short, None, KINDS, tag="cb1", rag_tag=None, today=date(2026, 11, 1)
        )


def test_run_takes_one_prompt_per_stage():
    argv = ["run", "--tag", "safety-v4-1", "--prompt", "grounded@v4", "--prompt", "partial@v4"]

    assert eval_answer._parse_args(argv).prompt == ["grounded@v4", "partial@v4"]


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--tag", "t1", "--prompt", "grounded@v3", "--prompt", "grounded@v4"],
        ["run", "--prompt", "grounded@v4", "--prompt", "partial@v4"],  # 꼬리표 없이는 받지 않는다
        ["compare", "--tag", "t1", "--prompt", "partial@v4"],
    ],
)
def test_run_rejects_bad_prompt_options(argv: list[str]):
    with pytest.raises(SystemExit):
        eval_answer._parse_args(argv)


def _write_measurement(tag: str, strata: list[AnswerRecord], verdict: str) -> None:
    """꼬리표 tag 의 F9 · 층화 기록과 층화 판정 파일을 지금 폴더 아래에 쓴다."""
    tagged = eval_answer.tagged
    for path in (eval_answer.ANSWERS_PATH, eval_answer.STRATA_ANSWERS_PATH):
        write_jsonl(tagged(path, tag), [record.to_json() for record in strata])
        meta = {"runs": list(_runs_with(["base@v4", f"grounded@{tag}"]))}
        tagged(path, tag).with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    graded = [record for record in strata if eval_answer.final_stage(record) == "grounded"]
    key = [{"judge_id": f"J{i}", "id": record.item.id} for i, record in enumerate(graded)]
    rows = [_verdict_row(row["judge_id"], verdict) for row in key]
    write_jsonl(tagged(eval_answer.JUDGE_KEY_PATH, tag), key)
    write_jsonl(tagged(eval_answer.VERDICTS_PATH, tag), rows)


def test_compare_with_a_base_tag_reads_that_tags_records_and_verdicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    strata = _strata_records()
    _write_measurement("grounded-v3", strata, "supported")
    _write_measurement("safety-v4-1", strata, "misread")
    base = eval_answer.Measurement.load("grounded-v3")
    other = eval_answer.Measurement.load("safety-v4-1")

    text = eval_answer.render_compare_report(
        base, other, tag="safety-v4-1", today=date(2026, 10, 31), base_tag="grounded-v3"
    )

    assert {verdict.verdict for verdict in base.verdicts.values()} == {"supported"}
    assert text.startswith("# grounded 프롬프트 비교 — 기준 `grounded-v3` vs `safety-v4-1`")
    assert "기준 = 꼬리표 `grounded-v3` 기록(data/eval/f9-answers-v1-grounded-v3.jsonl" in text
    assert "data/eval/f9-strata-verdicts-v1-grounded-v3.jsonl" in text
    assert "꼬리표 없는 기록" not in text
    assert "strata-report 리포트(high 확정)" not in text


def test_compare_without_a_base_tag_keeps_the_untagged_base_wording():
    strata = _strata_records()
    verdicts = {
        r.item.id: _verdict("J", "supported") for r in strata if r.stage_path == ("grounded",)
    }
    base = _measurement(strata, verdicts, [], ["grounded@v2"])

    text = eval_answer.render_compare_report(base, base, tag="v3", today=date(2026, 10, 29))

    assert text.startswith("# grounded 프롬프트 비교 — 기준 기록 vs `v3`")
    assert (
        "기준 = 꼬리표 없는 기록(data/eval/f9-answers-v1.jsonl · f9-strata-answers-v1.jsonl)"
        in text
    )
    assert "기준 기록의 판정은 strata-report 리포트(high 확정)에 쓴 것 그대로" in text


def test_cascade_verdicts_must_come_from_the_same_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tagged = eval_answer.tagged
    key = [{"judge_id": "C001", "id": "k1", "record": "f9", "side": "tag"}]
    verdicts = [_verdict_row("C001", "supported")]
    write_jsonl(tagged(eval_answer.CASCADE_VERDICTS_PATH, "t1"), verdicts)
    write_jsonl(tagged(eval_answer.CASCADE_KEY_PATH, "t1"), key)  # 기준이 없던 옛 열쇠

    assert set(eval_answer._read_cascade_verdicts("t1")) == {("f9", "tag", "k1")}
    with pytest.raises(SystemExit, match="cascade-input 부터 다시"):
        eval_answer._read_cascade_verdicts("t1", "grounded-v3")

    rows = [{**key[0], "base_tag": "grounded-v3"}]
    write_jsonl(tagged(eval_answer.CASCADE_KEY_PATH, "t1"), rows)

    assert set(eval_answer._read_cascade_verdicts("t1", "grounded-v3")) == {("f9", "tag", "k1")}
    with pytest.raises(SystemExit):
        eval_answer._read_cascade_verdicts("t1")


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--tag", "t1", "--base-tag", "grounded-v3"],  # compare · cascade-input 만
        ["compare", "--tag", "t1", "--base-tag", "t1"],
        ["compare", "--tag", "t1", "--base-tag", "Grounded_v3"],
    ],
)
def test_base_tag_is_checked(argv: list[str]):
    with pytest.raises(SystemExit):
        eval_answer._parse_args(argv)


def test_base_tag_parses_for_compare_and_cascade_input():
    for step in ("compare", "cascade-input"):
        argv = [step, "--tag", "safety-v4-1", "--base-tag", "grounded-v3"]

        assert eval_answer._parse_args(argv).base_tag == "grounded-v3"
