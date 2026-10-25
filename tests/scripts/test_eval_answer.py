"""scripts/eval_answer.py 의 표본 고르기 · 리포트를 합성 입력으로 — 데이터 · 모델 없이 돈다."""

import importlib.util
from dataclasses import replace
from datetime import date
from pathlib import Path

from playground_evalkit import KorQuADItem

from playground_wiki.evaluation import split_of
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem

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
