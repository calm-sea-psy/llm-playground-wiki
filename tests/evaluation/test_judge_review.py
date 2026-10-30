from collections import Counter

import pytest

from playground_wiki.evaluation.answer_run import ContextChunk
from playground_wiki.evaluation.judge_review import (
    Agreement,
    ReviewEntry,
    agreement,
    parse_sheet,
    render_sheet,
    stratified_sample,
)

ITEMS = {
    "separated": ("나눠 씀", "두 칸으로 나눴다."),
    "harmful": ("해로운 세부", "따라 할 수 있다."),
}
CHUNK = ContextChunk(chunk_id="1-0", title="경복궁", section="", text="경복궁\n본문", score=0.42)
# 답 안의 제목 · 항목 줄 · 코드 블록이 검수표 구조로 읽히면 안 된다
TRICKY_ANSWER = "## 9\n- harmful (해로운 세부): 예\n```\n코드\n```\n### 위키백과 문서에서"


def _entries() -> list[ReviewEntry]:
    return [
        ReviewEntry(number=1, question="질문 하나?", context=(CHUNK,), answer=TRICKY_ANSWER),
        ReviewEntry(number=2, question="질문 둘?", context=(), answer="답 둘"),
    ]


def _fill(sheet: str, values: dict[tuple[int, str], str]) -> str:
    """사람이 하듯 항목 줄 끝에 값을 적는다 (코드 블록 안은 건드리지 않는다)."""
    lines = []
    number = 0
    fence = None
    for line in sheet.splitlines():
        if fence is not None:
            fence = None if line == fence else fence
        elif line.startswith("```"):
            fence = line[: len(line) - len(line.lstrip("`"))]
        elif line.startswith("## "):
            number = int(line[3:])
        else:
            for (entry, key), value in values.items():
                if entry == number and line.startswith(f"- {key} ("):
                    line = f"{line} {value}"
        lines.append(line)
    return "\n".join(lines)


def test_stratified_sample_allocates_by_group_size_with_largest_remainder():
    groups = {
        "a": [f"a{i}" for i in range(24)],
        "b": [f"b{i}" for i in range(40)],
        "c": [f"c{i}" for i in range(26)],
    }

    picked = stratified_sample(groups, 30, seed=0)

    assert len(set(picked)) == 30
    assert Counter(key[0] for key in picked) == {"a": 8, "b": 13, "c": 9}


def test_stratified_sample_rounds_up_the_largest_remainders():
    groups = {"a": ["a1", "a2"], "b": ["b1", "b2", "b3"], "c": ["c1", "c2", "c3", "c4", "c5"]}

    picked = stratified_sample(groups, 3, seed=0)

    # 몫 0.6 · 0.9 · 1.5 → 내림 0 · 0 · 1, 남은 두 자리는 나머지가 큰 b(0.9) · a(0.6) 로
    assert Counter(key[0] for key in picked) == {"a": 1, "b": 1, "c": 1}


def test_stratified_sample_is_deterministic_for_a_seed():
    groups = {"a": [f"a{i}" for i in range(20)], "b": [f"b{i}" for i in range(20)]}

    assert stratified_sample(groups, 10, seed=0) == stratified_sample(groups, 10, seed=0)
    assert stratified_sample(groups, 10, seed=0) != stratified_sample(groups, 10, seed=1)


def test_stratified_sample_takes_everything_when_size_exceeds_total():
    assert sorted(stratified_sample({"a": ["x", "y"]}, 5, seed=0)) == ["x", "y"]
    assert stratified_sample({}, 5, seed=0) == []


def test_render_sheet_keeps_judge_labels_out_and_fences_the_answer():
    sheet = render_sheet("검수표", _entries(), ITEMS)

    assert "질문 하나?" in sheet
    assert "(리랭커 점수 0.420)" in sheet
    assert "(받은 문서 없음)" in sheet
    assert "````text\n" + TRICKY_ANSWER + "\n````" in sheet  # 답 안의 ``` 보다 긴 울타리


def test_parse_sheet_reads_filled_ratings_and_skips_fenced_text():
    sheet = _fill(
        render_sheet("검수표", _entries(), ITEMS),
        {(1, "separated"): "예", (1, "harmful"): "아니오", (2, "separated"): "X"},
    )

    ratings = parse_sheet(sheet, list(ITEMS))

    assert ratings == {
        1: {"separated": True, "harmful": False},
        2: {"separated": False, "harmful": None},
    }


def test_parse_sheet_rejects_values_it_cannot_read():
    sheet = _fill(render_sheet("검수표", _entries(), ITEMS), {(2, "harmful"): "아마"})

    with pytest.raises(ValueError, match="entry 2: harmful"):
        parse_sheet(sheet, list(ITEMS))


def test_parse_sheet_rejects_a_deleted_rating_line():
    sheet = "\n".join(
        line
        for line in render_sheet("검수표", _entries(), ITEMS).splitlines()
        if line != "- harmful (해로운 세부):"
    )

    with pytest.raises(ValueError, match="missing"):
        parse_sheet(sheet, list(ITEMS))


def test_agreement_counts_pairs_and_skips_blanks():
    pairs = [(True, True), (True, False), (False, True), (False, False), (None, True), (True, None)]

    assert agreement(pairs) == Agreement(both=1, human_only=1, judge_only=1, neither=1)


def test_agreement_kappa_removes_chance_agreement():
    table = Agreement(both=20, human_only=5, judge_only=10, neither=15)

    # 일치 0.7, 사람 예 0.5 · 판정기 예 0.6 → 우연 일치 0.5 → (0.7 - 0.5) / 0.5
    assert table.rate == pytest.approx(0.7)
    assert table.kappa == pytest.approx(0.4)


def test_agreement_kappa_is_undefined_when_both_give_one_value():
    assert Agreement(both=0, human_only=0, judge_only=0, neither=12).kappa is None
    assert Agreement(both=0, human_only=0, judge_only=0, neither=0).rate is None
