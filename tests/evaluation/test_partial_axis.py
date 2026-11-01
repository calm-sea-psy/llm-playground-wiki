"""partial 단계 블록 축(운영 대 공통) 짝 · 셈 · 리포트 — 합성 기록으로, 데이터 · 모델 없이."""

from dataclasses import replace
from datetime import date
from types import MappingProxyType

from playground_wiki.evaluation.answer_metrics import Share
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.partial_axis import (
    NOT_SCORED,
    ArmRun,
    AxisInputs,
    RoundPair,
    pair_cells,
    pair_records,
    render_report,
    two_section_format_rate,
)
from playground_wiki.evaluation.partial_judge import (
    DOC_HEADING,
    GENERAL_HEADING,
    NO_DIRECT,
    JudgedRow,
    PartialVerdict,
    check_format,
)

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"
REASON_MARK = "판정근거표지"
TWO_SECTIONS = f"{DOC_HEADING}\n{ANSWER_MARK}[1]\n{GENERAL_HEADING}\n알 수 없습니다."
EMPTY_DOC = f"{DOC_HEADING}\n{NO_DIRECT}. [1]\n{GENERAL_HEADING}\n{ANSWER_MARK}"
FREE_FORM = f"{ANSWER_MARK} 한 덩어리 답"
CHUNK = ContextChunk(chunk_id="1-0", title="문서", section="", text=CHUNK_MARK, score=0.5)
BASE = AnswerRecord(
    item=EvalItem(id="x", source="harmful", question=QUESTION_MARK, category="harmful"),
    stage_path=("partial",),
    answer=TWO_SECTIONS,
    hits=(CHUNK,),
    context_count=1,
    marker=None,
    cited=(1,),
    dropped=(),
    truncated=False,
    timed_out=False,
    error=None,
    top_score=0.5,
    gate_rule=None,
    spec_id="ollama:fake",
    prompt_version="p",
    search_ms=10.0,
    first_delta_ms=500.0,
    total_ms=2000.0,
    wall_ms=2000.0,
    done_reason={"partial": "stop"},
)
PROFILES = MappingProxyType(
    {"grounded": "base@v4+grounded@v5 · g", "partial": "base@v4+partial@v4 · p"}
)
COMMON_PROFILES = MappingProxyType(
    {"grounded": "base@v4+grounded@v5 · g", "partial": "base@v4 · p"}
)


def _record(
    item_id: str, *, stage: str = "partial", answer: str = TWO_SECTIONS, **item
) -> AnswerRecord:
    source = item.pop("source", "harmful")
    kind = item.pop("category", "harmful")
    record = replace(
        BASE,
        item=EvalItem(
            id=item_id, source=source, question=f"{QUESTION_MARK} {item_id}", category=kind, **item
        ),
        answer=answer,
        stage_path=(stage,) if stage != "partial_empty" else ("partial",),
    )
    if stage == "partial_empty":
        return replace(record, first_delta_ms=None, answer="고정 문구")
    if stage in ("gate", "refuse"):
        return replace(record, stage_path=(stage,), answer="고정 답")
    return record


def _korquad(item_id: str, **changes) -> AnswerRecord:
    return _record(
        item_id, source="korquad", category=None, gold_title="문서", gold_answer="정답", **changes
    )


def _offwiki(item_id: str, expected: str = "partial", **changes) -> AnswerRecord:
    return _record(item_id, source="offwiki", category=None, expected=expected, **changes)


def _judged(run: str, item_id: str, **verdict: bool) -> JudgedRow:
    values = {
        "doc_supported": True,
        "separated": True,
        "harmful": False,
        "refused": False,
    } | verdict
    return JudgedRow(
        run=run,
        id=item_id,
        judge="j",
        judge_prompt="p",
        format=check_format(""),
        verdict=PartialVerdict(**values, reason=REASON_MARK),
        error=None,
    )


def test_two_section_format_rate_counts_llm_partial_answers_only():
    records = [
        _record("a"),  # 두 칸 — 맞음
        _record(
            "b", answer=f"{DOC_HEADING}\n내용\n{GENERAL_HEADING}\n일반 지식 [2]"
        ),  # 일반 지식 칸 출처 번호
        _record("c", answer=FREE_FORM),  # 두 칸 제목 없음
        _record("d", stage="grounded"),  # partial 이 아니다
        _record("e", stage="partial_empty"),  # 고정 거절 문구
    ]

    assert two_section_format_rate(records) == Share(count=1, total=3)


def test_pair_records_keep_the_base_order_and_drop_one_sided_items():
    base = [_record("a"), _record("b"), _record("c")]
    common = [_record("c"), _record("a")]

    assert [(b.item.id, c.item.id) for b, c in pair_records(base, common)] == [
        ("a", "a"),
        ("c", "c"),
    ]


def test_pair_cells_count_usable_pairs_and_test_the_discordant_ones():
    pairs = [(True, False)] * 6 + [(True, True), (False, False), (None, True), (True, None)]

    cells = pair_cells(pairs)

    assert cells["짝"] == 8
    assert (cells["운영 예"], cells["공통 예"], cells["운영만"], cells["공통만"]) == (7, 1, 6, 0)
    assert cells["McNemar p"] == "0.0312 (검정력 낮음)"  # 2 × 0.5^6
    assert pair_cells([(None, None)])["McNemar p"] == "-"


def _inputs(**changes) -> AxisInputs:
    base_round = (
        _record("hz-01"),  # 두 칸 · 문서 칸 채움
        _record("hz-02", answer=EMPTY_DOC),
        _record("hz-03", stage="grounded"),
        _record("hc-01", category="contrast"),
    )
    common_round = (
        _record("hz-01", answer=FREE_FORM),
        _record("hz-02", answer=FREE_FORM),
        _record("hz-03", stage="grounded"),
        _record("hc-01", category="contrast", stage="grounded"),  # 생성 흔들림으로 단계가 갈림
    )
    procedure = (_record("hp-01", source="procedure", category="procedure"),)
    rounds = (
        RoundPair(
            "harmful",
            1,
            ArmRun("harmful-route-gate-v5-1", base_round, PROFILES),
            ArmRun("harmful-route-partial-base-1", common_round, COMMON_PROFILES),
        ),
        RoundPair(
            "procedure",
            1,
            ArmRun("procedure-route-gate-v5-1", procedure, PROFILES),
            ArmRun(
                "procedure-route-partial-base-1",
                (replace(procedure[0], answer=FREE_FORM),),
                COMMON_PROFILES,
            ),
        ),
    )
    f9_base = (
        _korquad("k1", answer=f"{TWO_SECTIONS} 정답"),
        _korquad("k2"),
        _korquad("k3", stage="grounded"),  # partial 이 아니라 대상 밖
        _offwiki("ow-001"),
        _offwiki("ow-002", expected="refuse", stage="refuse"),
    )
    f9_common = (
        _korquad("k1", answer=FREE_FORM),  # 정답을 잃음
        _korquad("k2", stage="grounded", answer="정답"),  # grounded 로 끝나 정답
        _offwiki("ow-001", answer=FREE_FORM),
        _offwiki("ow-002", expected="refuse", stage="refuse"),
    )
    judged = {
        ("f9-grounded-v5-1", "k1"): _judged("f9-grounded-v5-1", "k1"),
        ("f9-partial-base-1", "k1"): _judged("f9-partial-base-1", "k1", separated=False),
        ("procedure-route-gate-v5-1", "hp-01"): _judged("procedure-route-gate-v5-1", "hp-01"),
        ("procedure-route-partial-base-1", "hp-01"): _judged(
            "procedure-route-partial-base-1", "hp-01", refused=True
        ),
    }
    label = {"doc_supported": True, "separated": True, "harmful": False, "refused": False}
    values = {
        "rounds": rounds,
        "f9_base": ArmRun("f9-grounded-v5-1", f9_base, PROFILES),
        "f9_common": ArmRun("f9-partial-base-1", f9_common, COMMON_PROFILES),
        "f9_ids": ("k1", "k2", "ow-001"),
        "offwiki_ids": ("ow-001", "ow-002"),
        "judged": MappingProxyType(judged),
        "human_base": MappingProxyType(
            {
                ("harmful-route-gate-v5-1", "hz-01"): label,
                ("harmful-route-gate-v5-1", "hz-02"): label | {"refused": True},
                ("harmful-route-gate-v5-1", "hz-03"): label,  # grounded — 짝에서 빠진다
                ("harmful-route-gate-v5-1", "hc-01"): label,
            }
        ),
        "human_common": None,
        "kinds": MappingProxyType(
            {"hz-01": "harmful", "hz-02": "harmful", "hz-03": "harmful", "hc-01": "contrast"}
        ),
        "sheets": MappingProxyType({"운영": "base.md", "공통": "common.md"}),
    }
    return AxisInputs(**(values | changes))


def test_render_report_counts_without_question_answer_chunk_or_reason_text():
    text = render_report(_inputs(), today=date(2026, 11, 1))

    assert text.startswith("# partial 단계 블록 축 — 운영 대 공통 (2026-11-01)")
    for mark in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK, REASON_MARK):
        assert mark not in text
    assert "- 공통 칸 partial: base@v4 · p" in text
    assert "- grounded 설정이 답이 있는 모든 기록에서 같다" in text
    assert "harmful 회차 1 끝난 단계가 갈린 문항: hc-01 partial → grounded" in text
    # 문서 칸 비움: 운영 hz-02 만 — 공통은 두 칸 제목이 없어 아니오
    assert "| harmful | harmful | 운영 | 3 | 2 | 2/2 (100.0%) | 1/2 (50.0%) |" in text
    assert "| harmful | harmful | 공통 | 3 | 2 | 0/2 (0.0%) | 0/2 (0.0%) |" in text
    # 절차: 판정기 답하지 않음 짝 1 — 공통만 예
    assert "| 판정기 답하지 않음 (partial 답) | 1 | 1 | 0 | 1 | 0 | 1 | 1 (검정력 낮음) |" in text
    # F9: KorQuAD 대상 둘 — k1 은 정답을 잃고 k2 는 grounded 로 정답
    assert "| 정답 포함 | 2 | 1 | 1 | 1 | 1 | 1 (검정력 낮음) |" in text
    # 판정은 k1 만 있다 — 분모는 판정된 답, 괄호는 그 칸의 partial 답 (운영 k1 · k2 · ow-001, 공통 k1 · ow-001)
    assert (
        "| 나눠 씀 | **전체** | 1/1 (100.0%) (partial 답 3) | 0/1 (0.0%) (partial 답 2) |" in text
    )
    # 위키 밖: 기대 단계대로는 둘 다 같다
    assert "| **전체** | 2 | 2 | 2 | 2 | 0 | 0 | 1 (검정력 낮음) |" in text


def test_render_report_marks_human_cells_before_the_common_sheet_is_filled():
    before = render_report(_inputs(), today=date(2026, 11, 1))
    label = {"doc_supported": True, "separated": False, "harmful": False, "refused": True}
    after = render_report(
        _inputs(
            human_common=MappingProxyType(
                {
                    ("harmful-route-partial-base-1", "hz-01"): label,
                    ("harmful-route-partial-base-1", "hz-02"): label,
                }
            )
        ),
        today=date(2026, 11, 1),
    )

    assert f"| 해로운 세부 | harmful | 0/2 (0.0%) | {NOT_SCORED} (partial 답 2) |" in before
    # 채운 뒤: 짝 2 (hz-01 · hz-02 — hz-03 은 grounded 라 빠진다), 답하지 않음 운영 1 · 공통 2
    assert (
        "| 답하지 않음 | harmful | 1/2 (50.0%) | 2/2 (100.0%) | 1: 1/2 / 2/2 | 2 | 1 | 2 | 0 | 1 | 1 (검정력 낮음) |"
        in after
    )
    assert "| 나눠 씀 | harmful | 2/2 (100.0%) | 0/2 (0.0%) |" in after
    assert NOT_SCORED not in after.split("### 답 꼴")[0].split("### 사람 채점")[1]


def test_render_report_names_both_f9_records_when_the_operating_arm_was_rerun():
    default = render_report(_inputs(), today=date(2026, 11, 1))
    ops = (
        _korquad("k1", answer=f"{TWO_SECTIONS} 정답"),
        _korquad("k2", stage="grounded", answer="정답"),  # 다시 돌린 운영 칸도 흔들린다
        _offwiki("ow-001"),
        _offwiki("ow-002", expected="refuse", stage="refuse"),
    )
    rerun = render_report(
        _inputs(
            f9_base=ArmRun("f9-partial-ops-1", ops, PROFILES),
            f9_ids_run="f9-grounded-v5-1",
        ),
        today=date(2026, 11, 1),
    )

    assert "- 대상 = F9 운영 기록(`f9-grounded-v5-1`)에서 partial 로 끝난 3 문항" in default
    assert "표본 고정 기록" not in default
    assert (
        "- 대상 = 표본 고정 기록(`f9-grounded-v5-1`)에서 partial 로 끝난 3 문항. 운영 기록 `f9-partial-ops-1`"
        in rerun
    )
    # 두 칸의 끝난 단계 — 운영 k2 는 grounded
    assert "| 운영 | 3 | 0 | 0 | 1 | 2 | 0 | 0 |" in rerun
    assert "| 공통 | 3 | 0 | 0 | 1 | 2 | 0 | 0 |" in rerun
    # 정답 포함은 어느 단계든 — k1 운영만 · k2 둘 다
    assert "| 정답 포함 | 2 | 2 | 1 | 1 | 0 | 1 (검정력 낮음) |" in rerun
