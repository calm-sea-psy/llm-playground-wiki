"""scripts/eval_answer.py 의 표본 고르기 · 리포트를 합성 입력으로 — 데이터 · 모델 없이 돈다."""

import importlib.util
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from playground_corpus import Chunk
from playground_evalkit import KorQuADItem
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation import split_of
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.grounding import GroundingVerdict

_SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_answer.py"
_spec = importlib.util.spec_from_file_location("eval_answer", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
eval_answer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answer)

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"


def _paragraphs_in(split: str, count: int) -> list[str]:
    keys = (f"문서{i}#0" for i in range(1000))
    return [key for key in keys if split_of(key) == split][:count]


def _item(qid: str, title: str, answer: str) -> KorQuADItem:
    return KorQuADItem(id=qid, title=title, question=f"{qid}?", answer=answer, source_title=title)


def test_sample_rows_keep_eval_items_whose_answer_is_still_in_the_document():
    eval_a, eval_b, eval_c = _paragraphs_in("eval", 3)
    (bench,) = _paragraphs_in("bench", 1)
    items = [
        _item("a", "가", "정답"),  # 남음
        _item("b", "나", "옛 정답"),  # 문서가 고쳐져 정답 문장이 없다
        _item("c", "다", "정답"),  # 문서가 코퍼스에 없다
        _item("d", "가", "정답"),  # bench 몫
    ]
    paragraphs = {"a": eval_a, "b": eval_b, "c": eval_c, "d": bench}
    texts = {"가": "본문에 정 답이 있다", "나": "본문에 새 값만 있다"}
    offwiki = [
        {
            "id": "ow-001",
            "question": "q",
            "expected": "refuse",
            "category": "personal",
            "trap": False,
            "related_title": None,
            "note": "",
        }
    ]

    rows, picked = eval_answer.sample_rows(items, paragraphs, texts, offwiki)

    assert picked == 2  # a · b (c 는 문서가 없어 고르기 전에 빠진다)
    assert [(row["id"], row["source"]) for row in rows] == [("a", "korquad"), ("ow-001", "offwiki")]
    assert EvalItem.from_json(rows[0]).gold_answer == "정답"
    assert EvalItem.from_json(rows[1]).expected == "refuse"


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


def test_pool_items_keep_every_question_of_the_split_without_corpus_filters():
    eval_a, eval_b = _paragraphs_in("eval", 2)
    (bench,) = _paragraphs_in("bench", 1)
    items = [_item("c", "다", "정답"), _item("a", "가", "정답"), _item("b", "가", "옛 정답")]
    items.append(_item("d", "나", "정답"))
    paragraphs = {"a": eval_a, "b": eval_a, "c": eval_b, "d": bench}

    picked = eval_answer.pool_items(items, paragraphs, "eval")

    # 같은 문단의 질문 둘(a · b)도, 문서가 없는 질문(c)도 남는다
    assert [item.id for item in picked] == ["a", "b", "c"]


def test_pool_row_records_corpus_match_and_top_score():
    item = _item("a", "가", "정답")
    chunk = Chunk(
        id="1-0", doc_id="1", title="가", section="", text="가 > \n정답이다", seq=0, token_count=3
    )
    result = SearchResult(hits=(Hit(chunk=chunk, score=0.9),), took_ms=12.0)

    row = eval_answer.pool_row(item, "가#0", "본문에 새 값만 있다", result)
    empty = eval_answer.pool_row(item, "가#0", None, SearchResult(hits=(), took_ms=1.0))

    assert (row["in_corpus"], row["answer_in_document"]) == (True, False)
    assert (row["top_score"], row["answer_rank"]) == (0.9, 1)
    assert EvalItem.from_json(row).gold_answer == "정답"
    assert (empty["in_corpus"], empty["top_score"], empty["answer_rank"]) == (False, None, None)


def _pool_row(qid: str, paragraph: str, score: float | None) -> dict:
    return {
        "id": qid,
        "paragraph": paragraph,
        "top_score": score,
        "in_corpus": True,
        "answer_in_document": True,
        "answer_rank": 1,
    }


def _strata_pool() -> list[dict]:
    eval_a, eval_b = _paragraphs_in("eval", 2)
    bench_a, bench_b = _paragraphs_in("bench", 2)
    (calibrate,) = _paragraphs_in("calibrate", 1)
    return [
        _pool_row("e1", eval_a, 0.85),
        _pool_row("e2", eval_b, 0.90),
        _pool_row("e3", eval_a, 0.99),
        _pool_row("e4", eval_b, 0.98),
        _pool_row("e5", eval_b, 0.97),
        _pool_row("e6", eval_a, 0.50),  # 0.83 미만 — 고르지 않는다
        _pool_row("e7", eval_a, None),  # 관문 · 검색 결과 없음
        _pool_row("b1", bench_a, 0.88),
        _pool_row("b2", bench_b, 0.86),
        _pool_row("b3", bench_a, 0.99),
        _pool_row("b4", bench_b, 0.995),
        _pool_row("b5", bench_a, 0.96),
        _pool_row("c1", calibrate, 0.87),  # calibrate 몫은 쓰지 않는다
        _pool_row("c2", calibrate, 0.99),
    ]


def test_strata_rows_match_the_high_band_to_the_middle_band_per_split():
    picked = eval_answer.strata_rows(_strata_pool(), seed=0)

    ids = [row["id"] for row in picked]
    assert ids == sorted(ids)
    assert {"e1", "e2", "b1", "b2"} <= set(ids)
    assert len(set(ids) & {"e3", "e4", "e5"}) == 2  # eval 중간 2 → eval 높음 2
    assert len(set(ids) & {"b3", "b4", "b5"}) == 2  # bench 중간 2 → bench 높음 2
    assert not set(ids) & {"e6", "e7", "c1", "c2"}
    assert picked == eval_answer.strata_rows(_strata_pool(), seed=0)


def test_strata_rows_fill_the_middle_band_from_eval_first_and_keep_earlier_picks():
    rows = _strata_pool()

    two = {row["id"] for row in eval_answer.strata_rows(rows, seed=0, middle_size=2)}
    three = {row["id"] for row in eval_answer.strata_rows(rows, seed=0, middle_size=3)}

    assert len(two) == 4
    assert {"e1", "e2"} <= two  # eval 중간을 먼저 다 담는다
    assert not two & {"b1", "b2", "b3", "b4", "b5"}
    assert len(three) == 6
    assert two < three  # 늘려도 앞서 고른 문항이 남는다
    assert len(three & {"b1", "b2"}) == 1
    assert len(three & {"b3", "b4", "b5"}) == 1


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
    return GroundingVerdict.from_json(
        {
            "judge_id": judge_id,
            "verdict": verdict,
            "claim": f"주장 {ANSWER_MARK}",
            "evidence": f"[1] {CHUNK_MARK}",
            "reason": "이유",
            "wrong_citation": False,
            "off_question": False,
        }
    )


def test_judge_rows_hide_score_band_gold_and_item_id():
    records = _strata_records()

    rows, key = eval_answer.judge_rows(records, seed=0)

    # grounded 로 끝난 m1 · m2 · h1 · h2 만 (m3 은 못 찾음 → partial)
    assert sorted(row["id"] for row in key) == ["h1", "h2", "m1", "m2"]
    assert [row["judge_id"] for row in rows] == ["J001", "J002", "J003", "J004"]
    assert [row["judge_id"] for row in key] == ["J001", "J002", "J003", "J004"]
    assert set(rows[0]) == {"judge_id", "question", "documents", "answer"}
    assert rows[0]["documents"] == [{"number": 1, "text": f"흥선대원군 {CHUNK_MARK}"}]
    text = json.dumps(rows, ensure_ascii=False)
    for hidden in ("0.85", "0.99", '흥선대원군"', "m1", "h1"):
        assert hidden not in text


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


def test_strata_report_needs_a_verdict_for_every_grounded_answer():
    records = _strata_records()
    pool_rows = [_pool_row(record.item.id, "문서#0", 0.9) for record in records]

    with pytest.raises(ValueError, match="have no verdict"):
        eval_answer.render_strata_report(records, RUNS, pool_rows, {}, today=date(2026, 10, 29))


def test_tagged_appends_the_tag_to_the_file_name_and_keeps_the_base_path():
    path = Path("data/eval/f9-answers-v1.jsonl")

    assert eval_answer.tagged(path, None) == path
    assert eval_answer.tagged(path, "grounded-v3") == Path(
        "data/eval/f9-answers-v1-grounded-v3.jsonl"
    )
    meta = eval_answer.tagged(path, "grounded-v3").with_suffix(".meta.json")
    assert meta.name == "f9-answers-v1-grounded-v3.meta.json"


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


def test_partial_sections_split_the_document_part_from_general_knowledge():
    doc, general = eval_answer.PARTIAL_DOC_HEADING, eval_answer.PARTIAL_GENERAL_HEADING

    assert eval_answer.partial_sections(f"{doc}\n가 [1].\n\n{general}\n나") == ("가 [1].", "나")
    assert eval_answer.partial_sections(f"{doc}\n가 [1].") == ("가 [1].", "")
    assert eval_answer.partial_sections(f"가 [1].\n{general}\n나") == ("가 [1].", "나")


def _cascade_measurements() -> tuple[object, object]:
    doc, general = eval_answer.PARTIAL_DOC_HEADING, eval_answer.PARTIAL_GENERAL_HEADING
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


def test_cascade_rows_judge_only_the_document_part_and_hide_which_record():
    base, other = _cascade_measurements()

    rows, key = eval_answer.cascade_rows(base, other, seed=0)

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
