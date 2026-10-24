import random
from dataclasses import replace

import numpy as np
import pytest
from playground_corpus import Chunk
from playground_evalkit import KorQuADItem
from playground_retrieval.core import Hit

from playground_wiki.evaluation.answer_metrics import (
    FINAL_STAGES,
    LATENCY_PERCENTILES,
    SCORE_BAND_EDGES,
    PairedHits,
    Share,
    answer_rank,
    answered,
    cites_gold,
    contains_gold,
    final_stage,
    gold_in_hits,
    gold_numbers,
    has_dropped,
    latency,
    marker_mismatch,
    marker_missing,
    matches_expected,
    mcnemar_exact,
    paired_hits,
    path_counts,
    percentile,
    says_no_info,
    score_band,
    score_bands,
    share,
    stage_counts,
    stage_path,
)
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem

PALACE = EvalItem(
    id="k1",
    source="korquad",
    question="경복궁을 중건한 사람은?",
    gold_title="경복궁",
    gold_answer="흥선대원군",
)
PERSONAL = EvalItem(
    id="o1", source="offwiki", question="내 월급날은?", expected="refuse", category="personal"
)
AFTER_DUMP = EvalItem(
    id="o2",
    source="offwiki",
    question="2027년 우승팀은?",
    expected="partial",
    category="after_dump",
)
GOLD = ContextChunk("1-0", "경복궁", "역사", "경복궁 > 역사\n흥선 대원군이 중건했다.", 0.97)
OTHER = ContextChunk("2-0", "창덕궁", "", "창덕궁 > \n태종이 지었다.", 0.91)
WRONG_DOC = ContextChunk("3-0", "운현궁", "", "운현궁 > \n흥선대원군이 살았다.", 0.88)
BASE = AnswerRecord(
    item=PALACE,
    stage_path=("grounded",),
    answer="흥선대원군이 중건했다 [1].",
    hits=(GOLD, OTHER),
    context_count=2,
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
)


def test_share_counts_true_among_answers_that_apply():
    result = share([True, False, None, True])

    assert result == Share(count=2, total=3)
    assert result.rate == pytest.approx(2 / 3)
    assert share([None]).rate is None


@pytest.mark.parametrize(
    ("path", "error", "final", "name"),
    [
        (("grounded",), None, "grounded", "grounded"),
        (("grounded", "partial"), None, "partial", "grounded → partial"),
        (("gate",), None, "gate", "gate"),
        (("grounded",), "generation", "error", "grounded → error"),
        (("grounded", "partial"), "timeout", "error", "grounded → partial → error"),
    ],
)
def test_final_stage_and_stage_path(path, error, final, name):
    record = replace(BASE, stage_path=path, error=error)

    assert (final_stage(record), stage_path(record)) == (final, name)


def test_path_counts_are_sorted_by_count_then_name():
    records = [
        replace(BASE, stage_path=("partial",)),
        BASE,
        replace(BASE, stage_path=("grounded", "partial")),
        BASE,
    ]

    assert list(path_counts(records).items()) == [
        ("grounded", 2),
        ("grounded → partial", 1),
        ("partial", 1),
    ]


def test_stage_counts_follow_the_column_order_with_zeros():
    records = [BASE, replace(BASE, stage_path=("refuse",)), replace(BASE, error="timeout")]

    counts = stage_counts(records)

    assert list(counts) == list(FINAL_STAGES)
    assert counts == {"gate": 0, "refuse": 1, "grounded": 1, "partial": 0, "error": 1}


def test_contains_gold_ignores_spaces_and_skips_items_without_gold():
    assert contains_gold(replace(BASE, answer="흥선 대원군이 중건했다.")) is True
    assert contains_gold(replace(BASE, answer="고종이 중건했다.")) is False
    assert contains_gold(replace(BASE, item=PERSONAL)) is None


def test_gold_numbers_are_context_chunks_of_the_gold_document_with_the_answer():
    record = replace(BASE, hits=(OTHER, WRONG_DOC, GOLD), context_count=3)

    assert gold_numbers(record) == (3,)  # 운현궁 청크는 정답 문자열이 있어도 정답 문서가 아니다
    assert gold_numbers(replace(record, context_count=2)) == ()
    assert gold_numbers(replace(record, item=PERSONAL)) == ()


def test_gold_in_hits_looks_beyond_the_context():
    record = replace(BASE, hits=(OTHER, GOLD), context_count=1)

    assert gold_in_hits(record) is True
    assert gold_in_hits(replace(record, hits=(OTHER,))) is False
    assert gold_in_hits(replace(record, item=PERSONAL)) is None


def test_cites_gold_needs_a_gold_chunk_in_the_context():
    assert cites_gold(BASE) is True
    assert cites_gold(replace(BASE, cited=(2,))) is False
    assert cites_gold(replace(BASE, cited=())) is False
    assert cites_gold(replace(BASE, hits=(OTHER, GOLD), context_count=1)) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("문서에는 석굴암의 무게에 대한 정보는 없습니다.", True),
        ("주어진 문서에서 답을 찾을 수 없습니다.", True),
        ("2025년 MVP 는 문서에 언급되어 있지 않습니다.", True),
        ("해당 내용은 제공된 자료에 나와 있지 않습니다.", True),
        ("문서에서 확인할 수 없습니다.", True),
        ("흥선대원군이 중건했다 [1].", False),
        ("최근 값은 알 수 없으나 2026년 10월 1일 기준 1,000원입니다 [1].", False),
        ("명왕성은 2006년부터 행성에 포함되지 않는다 [1].", False),
        ("정보통신부가 없앴다.", False),
    ],
)
def test_says_no_info(text, expected):
    assert says_no_info(text) is expected


def test_marker_missing_counts_only_records_whose_marker_was_read():
    assert marker_missing(replace(BASE, marker="missing")) is True
    assert marker_missing(replace(BASE, marker="not_found")) is False
    assert marker_missing(replace(BASE, marker=None, stage_path=("partial",))) is None


def test_marker_mismatch_is_found_marker_with_a_no_info_body():
    no_info = replace(BASE, answer="문서에 관련 정보가 없습니다.")

    assert marker_mismatch(no_info) is True
    assert marker_mismatch(BASE) is False
    # 찾음 · 빈 본문으로 캐스케이드한 것은 partial 답이라 보지 않는다
    assert marker_mismatch(replace(no_info, stage_path=("grounded", "partial"))) is None
    assert marker_mismatch(replace(no_info, marker="missing")) is None
    assert marker_mismatch(replace(no_info, error="timeout")) is None


def test_has_dropped_only_for_llm_answers():
    assert has_dropped(replace(BASE, dropped=(7,))) is True
    assert has_dropped(BASE) is False
    assert has_dropped(replace(BASE, stage_path=("gate",))) is None
    assert answered(replace(BASE, stage_path=("refuse",))) is False


@pytest.mark.parametrize(
    ("item", "path", "expected"),
    [
        (PERSONAL, ("gate",), True),
        (PERSONAL, ("refuse",), True),
        (PERSONAL, ("partial",), False),
        (AFTER_DUMP, ("grounded", "partial"), True),
        (AFTER_DUMP, ("grounded",), False),
        (PALACE, ("grounded",), None),
    ],
)
def test_matches_expected(item, path, expected):
    assert matches_expected(replace(BASE, item=item, stage_path=path)) is expected


def test_score_bands_split_at_the_edges():
    assert SCORE_BAND_EDGES == (0.83, 0.94)
    assert [score_band(s) for s in (0.5, 0.83, 0.93, 0.94, 0.99, None)] == [
        "< 0.83",
        "0.83–0.94",
        "0.83–0.94",
        "≥ 0.94",
        "≥ 0.94",
        "없음",
    ]
    assert score_bands() == ("< 0.83", "0.83–0.94", "≥ 0.94", "없음")


def test_percentile_matches_numpy():
    rng = random.Random(0)
    values = [rng.uniform(0, 5000) for _ in range(37)]

    for q in (0, 50, 90, 95, 99, 100):
        assert percentile(values, q) == pytest.approx(np.percentile(values, q))
    assert percentile([10.0, 20.0, 30.0, 40.0], 90) == pytest.approx(37.0)
    assert percentile([], 50) is None
    with pytest.raises(ValueError, match="q must be"):
        percentile(values, 101)


def test_latency_skips_missing_times():
    result = latency([100.0, None, 300.0])

    assert result.count == 2
    assert list(result.percentiles) == list(LATENCY_PERCENTILES)
    assert result.percentiles[50] == pytest.approx(200.0)
    assert latency([None]).percentiles == {}


def test_answer_rank_is_the_first_gold_chunk():
    item = KorQuADItem(
        id="k1", title="경복궁", question="q", answer="흥선대원군", source_title="경복궁"
    )
    hits = [
        Hit(chunk=_chunk("운현궁", "흥선대원군이 살았다."), score=0.9),
        Hit(chunk=_chunk("경복궁", "흥선 대원군이 중건했다."), score=0.8),
    ]

    assert answer_rank(hits, item) == 2
    assert answer_rank(hits[:1], item) is None


def _chunk(title: str, text: str) -> Chunk:
    return Chunk(id="1-0", doc_id="1", title=title, section="", text=text, seq=0, token_count=4)


def test_paired_hits_count_each_cell():
    table = paired_hits([(True, True), (True, False), (True, False), (False, True), (False, False)])

    assert table == PairedHits(both=1, original_only=2, paraphrase_only=1, neither=1)
    assert table.total == 5
    assert (table.original_rate, table.paraphrase_rate) == (0.6, 0.4)
    assert table.mcnemar_p() == mcnemar_exact(2, 1)
    assert paired_hits([]).original_rate is None


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (1, 9, 22 / 1024),  # 2 × (C(10,0) + C(10,1)) / 2^10
        (9, 1, 22 / 1024),
        (0, 6, 2 / 64),
    ],
)
def test_mcnemar_exact(first, second, expected):
    assert mcnemar_exact(first, second) == pytest.approx(expected)
