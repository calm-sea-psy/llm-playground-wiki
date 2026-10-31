import json

import pytest
from playground_llm.fake import FakeLLM

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.partial_judge import (
    DOC_HEADING,
    GENERAL_HEADING,
    ITEM_GUIDE,
    JUDGE_ITEMS,
    JUDGE_SPEC,
    NO_DIRECT,
    SCHEMA,
    SYSTEM,
    UNKNOWN,
    JudgedRow,
    check_format,
    format_documents,
    judge_partial,
    judged_row,
    parse_verdict,
    sections,
)
from playground_wiki.generation import PROMPT_DIR

CHUNK = ContextChunk(
    chunk_id="1-0",
    title="경복궁",
    section="",
    text="경복궁 > 역사\n흥선대원군이 중건했다.",
    score=0.4,
)
GOOD = (
    f"{DOC_HEADING}\n- 흥선대원군이 중건했다[1].\n\n"
    f"{GENERAL_HEADING}\n- 중건 비용은 확실하지 않습니다."
)
VERDICT = {
    "reason": "문서 칸이 [1] 과 맞고 일반 지식 칸에 번호가 없다.",
    "doc_supported": True,
    "separated": True,
    "harmful": False,
    "refused": False,
}


def test_headings_and_phrases_match_partial_prompt():
    prompt = (PROMPT_DIR / "partial" / "v2.md").read_text(encoding="utf-8")

    for text in (DOC_HEADING, GENERAL_HEADING, NO_DIRECT, UNKNOWN):
        assert text in prompt


def test_item_guide_and_schema_cover_every_judge_item():
    assert tuple(ITEM_GUIDE) == JUDGE_ITEMS
    assert SCHEMA["required"] == ["reason", *JUDGE_ITEMS]
    for name in JUDGE_ITEMS:
        assert f"{name}: {ITEM_GUIDE[name][1]}" in SYSTEM


def test_sections_split_by_heading_lines_and_drop_text_before_them():
    parts = sections(f"머리말\n  {DOC_HEADING}  \n문서 내용[1]\n{GENERAL_HEADING}\n일반 지식")

    assert parts == {DOC_HEADING: "문서 내용[1]", GENERAL_HEADING: "일반 지식"}


def test_sections_join_lines_until_the_next_heading():
    assert sections(f"머리\n{DOC_HEADING}\n가\n {GENERAL_HEADING} \n나\n다") == {
        DOC_HEADING: "가",
        GENERAL_HEADING: "나\n다",
    }


def test_sections_do_not_treat_inline_heading_text_as_a_heading():
    assert sections(f"앞에 글이 있는 {DOC_HEADING} 줄") == {}


def test_check_format_accepts_two_sections_with_citations_only_in_doc():
    result = check_format(GOOD)

    assert result.well_formed
    assert not result.general_cites
    assert not result.doc_no_direct
    assert not result.general_unknown


def test_check_format_flags_citation_in_general_section():
    result = check_format(f"{DOC_HEADING}\n{NO_DIRECT}.\n{GENERAL_HEADING}\n흥선대원군[2]")

    assert result.general_cites
    assert result.doc_no_direct
    assert not result.well_formed


def test_check_format_flags_missing_general_section():
    result = check_format(f"{DOC_HEADING}\n흥선대원군이 중건했다[1].")

    assert result.has_doc
    assert not result.has_general
    assert not result.general_cites
    assert not result.well_formed


def test_check_format_reads_unknown_phrase_in_general_section():
    result = check_format(f"{DOC_HEADING}\n{NO_DIRECT}.\n{GENERAL_HEADING}\n{UNKNOWN}.")

    assert result.general_unknown
    assert result.well_formed


def test_format_documents_numbers_chunks_like_the_answer_path():
    other = ContextChunk(chunk_id="2-0", title="창덕궁", section="", text="창덕궁 본문", score=0.3)

    text = format_documents([CHUNK, other])

    assert text.index('<document id="1" title="경복궁">') < text.index(
        '<document id="2" title="창덕궁">'
    )
    assert CHUNK.text in text


def test_format_documents_says_so_when_there_is_no_context():
    assert format_documents([]) == "(받은 문서 없음)"


def test_judge_partial_sends_question_documents_and_answer_with_schema():
    llm = FakeLLM([json.dumps(VERDICT, ensure_ascii=False)])

    verdict = judge_partial(llm, "경복궁을 중건한 사람은?", GOOD, [CHUNK])

    call = llm.calls[0]
    assert call.spec == JUDGE_SPEC
    assert call.json_schema == SCHEMA
    assert call.messages[0].content == SYSTEM
    user = call.messages[1].content
    assert user.index("경복궁을 중건한 사람은?") < user.index(CHUNK.text) < user.index(GOOD)
    assert verdict.separated
    assert not verdict.harmful
    assert verdict.reason == VERDICT["reason"]
    assert [verdict.item(name) for name in JUDGE_ITEMS] == [True, True, False, False]


def test_parse_verdict_rejects_text_that_is_not_json():
    with pytest.raises(ValueError, match="not JSON"):
        parse_verdict("판정: 해롭지 않음")


def test_parse_verdict_rejects_missing_or_non_boolean_items():
    with pytest.raises(ValueError, match="booleans"):
        parse_verdict(json.dumps({**VERDICT, "harmful": "false"}))
    with pytest.raises(ValueError, match="booleans"):
        parse_verdict(json.dumps({k: v for k, v in VERDICT.items() if k != "refused"}))


# judged_row 시험 (scripts/judge_partial.py 에서 옮겨 옴) — 위 VERDICT 와 이름이 겹쳐 판정은 JUDGED_VERDICT
QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"
REASON_MARK = "판정근거표지"
BASE = AnswerRecord(
    item=EvalItem(id="x", source="offwiki", question=f"{QUESTION_MARK}?", expected="partial"),
    stage_path=("partial",),
    answer=f"{DOC_HEADING}\n{ANSWER_MARK}[1]\n{GENERAL_HEADING}\n알 수 없습니다.",
    hits=(
        ContextChunk(
            chunk_id="1-0", title="문서", section="", text=f"문서\n{CHUNK_MARK}", score=0.5
        ),
    ),
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
    prompt_version="base@v4:x+partial@v2:y+question@v1:z",
    search_ms=50.0,
    first_delta_ms=900.0,
    total_ms=2000.0,
    wall_ms=2001.0,
    done_reason={"partial": "stop"},
)
JUDGED_VERDICT = {
    "doc_supported": True,
    "separated": True,
    "harmful": False,
    "refused": True,
    "reason": REASON_MARK,
}


def test_judged_row_keeps_format_and_verdict():
    llm = FakeLLM([json.dumps(JUDGED_VERDICT, ensure_ascii=False)])

    row = judged_row("f9", BASE, llm)

    assert (row.run, row.id, row.error) == ("f9", "x", None)
    assert row.format.has_doc is True
    assert row.verdict is not None
    assert row.verdict.refused is True
    assert CHUNK_MARK in llm.calls[0].messages[1].content


def test_judged_row_records_a_bad_reply_as_error():
    row = judged_row("f9", BASE, FakeLLM(["판정할 수 없습니다"]))

    assert row.verdict is None
    assert "not JSON" in str(row.error)


def test_judged_row_json_keeps_the_key_order_and_reads_back():
    judged = judged_row("f9", BASE, FakeLLM([json.dumps(JUDGED_VERDICT, ensure_ascii=False)]))
    failed = judged_row("f9", BASE, FakeLLM(["판정할 수 없습니다"]))

    row = judged.to_json()

    assert list(row) == ["run", "id", "judge", "judge_prompt", "format", "verdict", "error"]
    assert list(row["verdict"]) == ["doc_supported", "separated", "harmful", "refused", "reason"]
    assert failed.to_json()["verdict"] is None
    for item in (judged, failed):
        assert JudgedRow.from_json(item.to_json()) == item
