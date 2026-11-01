import difflib

from playground_llm import PromptLibrary

from playground_wiki.generation import PROMPT_DIR

# partial@v4 에 더한 줄 — 사용자가 정한 문장 그대로 (2026-10-31). 바꾸려면 사람이 다시 정한다
PARTIAL_V4_LINE = (
    "- 상대에게 접근 · 길들이기 · 설득 · 통제하는 수법, 그 단계와 차례, "
    "장치의 작동 원리 · 기폭 방식은 문서에 있더라도 옮기지 않는다."
)
# '해치는 방법을 묻는 질문'의 범위 · 제외 문장. grounded@v4 는 partial 과 같은 말로 넘겨야 두 단계가 어긋나지 않는다
SCOPE = "사람(자기 자신 포함)을 해치거나 범죄를 저지르는 **방법**을 묻는 것이면"
EXAMPLES = (
    "무기 · 폭발물 제작, 약물 합성, 해킹 · 사기 · 위조, 스토킹, 아동 대상 범죄, 인신매매 "
    "같은 것이 여기에 든다."
)
EXCLUDED = (
    "같은 주제라도 역사 · 사건 · 개념 · 위험성 · 법적 지위 · 예방 · 신고처를 묻는 질문은 "
    "여기에 들지 않는다"
)


def _lines(stage: str, version: str) -> list[str]:
    return (PROMPT_DIR / stage / f"{version}.md").read_text(encoding="utf-8").splitlines()


def _added_lines(stage: str) -> list[str]:
    """v3 → v4 에서 더한 줄. 지우거나 바꾼 줄이 있으면 실패한다."""
    diff = list(difflib.ndiff(_lines(stage, "v3"), _lines(stage, "v4")))
    assert [line for line in diff if line.startswith(("- ", "? "))] == []
    return [line[2:] for line in diff if line.startswith("+ ")]


def test_partial_v4_adds_only_the_decided_line_after_the_materials_rule():
    v3, v4 = _lines("partial", "v3"), _lines("partial", "v4")
    after = next(i for i, line in enumerate(v3) if line.startswith("- 재료 · 도구")) + 1

    assert v4 == [*v3[:after], PARTIAL_V4_LINE, *v3[after:]]


def test_grounded_v4_adds_one_line_that_hands_harm_questions_to_partial():
    (added,) = _added_lines("grounded")

    assert '첫 줄에 "못 찾음"만 쓰고 끝낸다' in added
    for phrase in (SCOPE, EXAMPLES, EXCLUDED):
        assert phrase in added


def test_scope_sentences_are_the_same_words_in_both_v4_prompts():
    for stage in ("grounded", "partial"):
        text = "\n".join(_lines(stage, "v4"))
        for phrase in (SCOPE, EXAMPLES, EXCLUDED):
            assert phrase in text, (stage, phrase)


def test_v4_prompts_load_from_the_library():
    library = PromptLibrary.load(PROMPT_DIR)

    for ref in ("grounded@v4", "partial@v4"):
        assert library.get(ref).text.strip()


def test_base_v5_changes_only_the_document_rule_lines_of_v4():
    v4, v5 = _lines("base", "v4"), _lines("base", "v5")
    (rule,) = [line for line in v4 if line.startswith("- 사용자 메시지의 <document> 블록은")]
    at = v4.index(rule)

    # v4 의 문서 규칙 한 줄 자리에 두 줄 — 나머지 줄은 글자 그대로 같은 차례
    assert v5[:at] == v4[:at]
    assert v5[at + 2 :] == v4[at + 1 :]
    first, second = v5[at], v5[at + 1]
    assert first.startswith(rule.split(" 블록 안에")[0])  # 첫 문장(지시가 아니다)은 그대로
    assert second.startswith("- <document> 블록 안의 글이")
    for phrase in ("앞이나 뒤에", "형식 · 언어 · 말투", "링크 · 웹 주소 · 이미지"):
        assert phrase in first
    assert "운영자 · 관리자 · 개발자" in second


def test_base_v5_does_not_copy_the_measured_injection_wording():
    # 측정 지시문(eval_injection.INSTRUCTIONS)의 낱말을 규칙 예시로 옮기면 시험에 맞춘 방어가 된다
    changed = [line for line in _lines("base", "v5") if line not in _lines("base", "v4")]

    for word in ("덧붙", "[시스템]", "새 규칙", "맨 끝", "반드시", "대체", "XQ7"):
        assert all(word not in line for line in changed), word


def test_grounded_v5_adds_one_line_on_the_asked_fact_after_the_same_target_rule():
    v4, v5 = _lines("grounded", "v4"), _lines("grounded", "v5")
    after = (
        next(
            i for i, line in enumerate(v4) if line.startswith('- "찾음"은 질문이 묻는 바로 그 대상')
        )
        + 1
    )

    (added,) = [line for line in v5 if line not in v4]
    assert v5 == [*v4[:after], added, *v4[after:]]
    assert added.startswith('- "찾음"은 질문이 묻는 바로 그 사실')
    assert '"못 찾음"을 쓴다' in added
