"""반사실 문항 검사 — 원래 값이 다른 꼴로 남거나 값이 문맥과 어긋나는 문항을 고른다."""

import re
from collections.abc import Sequence
from itertools import chain, combinations

from playground_evalkit import squash

from playground_wiki.evaluation.counterfactual_rules import (
    ANNIVERSARY,
    ARITHMETIC,
    BOUNDED,
    COMPUTED_OPERAND_MAX,
    COUNT_MIN,
    COUNTERFACTUAL_EXCLUDE,
    DIGITS,
    ERA_YEAR,
    MONTH_DAY,
    NAME_PART_MIN,
    NUMBER_LEFT_MIN,
    ORIGINAL_SCRIPT,
    PERSON_CUE,
    PERSON_CUE_WINDOW,
    PERSON_QUESTION,
    PERSON_REPLACE_MAX,
    QUESTION_YEAR,
    RANGE_AFTER,
    RANGE_BEFORE,
    RANGE_BETWEEN,
    RANGE_GAP,
    RELATIVE_WINDOW,
    RELATIVE_YEAR,
    ROLE_SUFFIXES,
    SPOT_WIDTH,
    YEAR_IN_ANSWER,
    Passage,
    SourceItem,
    answer_windows,
    context_sentences,
    flexible_pattern,
    is_name,
    keeps_order,
    name_style,
    run_kinds,
    run_pattern,
)


def is_counterfactual_candidate(item: SourceItem) -> bool:
    """반사실 후보인지: 정답 표기가 하나이고 날짜 · 숫자 · 인물 꼴."""
    return len(item.answers) == 1 and _is_counterfactual_shape(item)


def document_years(source: SourceItem, answer: str) -> list[int]:
    """정답 문서 청크 전체에 나온 다른 연도들 — 연도 답은 이들과의 앞뒤 순서를 지킨다.

    답 근처(NEIGHBOR_WINDOW)만 보면 같은 청크 멀리 있는 '애틀랜타에서 부진한 뒤 … 1997년 실내 시즌' 을 못 봐
    애틀랜타 올림픽을 1998년으로 옮겼다. 정답 표기 자리와 답 자신의 연도는 뺀다.
    """
    own = {int(year) for year in YEAR_IN_ANSWER.findall(answer)}
    if not own:
        return []
    pattern = flexible_pattern(answer)
    year_pattern = run_pattern("year")
    years: list[int] = []
    for chunk in source.context:
        if chunk.title != source.gold_title:
            continue
        masked = pattern.sub(" ", chunk.text)
        years += [int(year) for year in year_pattern.findall(masked) if int(year) not in own]
    return years


def document_dates(source: SourceItem, answer: str) -> list[tuple[int, int]]:
    """연도 없는 월일 답이면 정답 문서 청크 전체에 나온 다른 (월, 일)들 — 답은 이들과의 앞뒤 순서를 지킨다.

    사건을 날짜순으로 적은 문서('6월 1일 볼 … 6월 11일 호킨스')에서 6월 16일로 옮기면 순서가 뒤집힌다.
    연도가 든 답(연도 순서는 document_years 가 본다)이나 월일이 없는 답이면 빈 목록. 정답 표기 자리는 뺀다.
    """
    if YEAR_IN_ANSWER.search(answer) or not MONTH_DAY.search(answer):
        return []
    pattern = flexible_pattern(answer)
    dates: list[tuple[int, int]] = []
    for chunk in source.context:
        if chunk.title != source.gold_title:
            continue
        masked = pattern.sub(" ", chunk.text)
        dates += [(int(month), int(day)) for month, day in MONTH_DAY.findall(masked)]
    return dates


def keeps_date_order(answer: str, new: str, doc_dates: Sequence[tuple[int, int]]) -> bool:
    """New 의 (월, 일)이 answer 의 (월, 일)처럼 문서의 다른 월일들과 앞뒤 순서를 지키고 겹치지 않는지."""
    old_match, new_match = MONTH_DAY.search(answer), MONTH_DAY.search(new)
    if old_match is None or new_match is None:
        return True
    old = (int(old_match[1]), int(old_match[2]))
    moved = (int(new_match[1]), int(new_match[2]))
    return all(
        (old > other) == (moved > other) and moved != other for other in doc_dates if other != old
    )


def keeps_year_order(answer: str, new: str, doc_years: Sequence[int]) -> bool:
    """New 의 연도가 answer 의 연도 자리마다 문서의 다른 연도들과 앞뒤 순서를 지키는지. 연도가 없으면 True."""
    old_years = [int(year) for year in YEAR_IN_ANSWER.findall(answer)]
    new_years = [int(year) for year in YEAR_IN_ANSWER.findall(new)]
    return len(old_years) == len(new_years) and all(
        keeps_order([old], new_year - old, doc_years)
        for old, new_year in zip(old_years, new_years, strict=True)
    )


def precheck(source: SourceItem, answer: str) -> str | None:
    """바꾸기 전에 거르는 규칙 (2026-10-28 사람 검수 v1 · v2 20문항씩에서 나온 꼴). 걸리면 이유, 아니면 None."""
    if source.item_id.split(":", 1)[-1] in COUNTERFACTUAL_EXCLUDE:
        return "excluded_by_review"
    context = source.context
    pattern = flexible_pattern(answer)
    if any(squash(answer) in squash(chunk.title) for chunk in context):
        return "answer_in_title"  # 청크 제목은 프롬프트에 그대로 실린다 — 원래 값이 새어 나간다
    windows = list(chain.from_iterable(answer_windows(chunk.text, pattern) for chunk in context))
    if not windows:
        return "answer_not_in_context"
    person = source.answer_shape == "person"
    # 수 · 연도 없는 날짜(800만 · 18일)는 다른 문서에서 다른 사실이기 쉽다 — 바꾸면 그 문서의 사실(노트8 의
    # 800만 화소)이 틀어지고, 두면 원래 값이 남는다. 인물 · 연도는 다른 문서에서도 대개 같은 사실이라 함께 바꾼다
    if (
        not person
        and not YEAR_IN_ANSWER.search(answer)
        and any(pattern.search(chunk.text) for chunk in context if chunk.title != source.gold_title)
    ):
        return "answer_in_other_document"
    if _is_yearless_date(answer) and len(_distinct_spots(context, pattern)) > 1:
        # 연도 없는 날짜가 두 문장 이상에 나온다 — 다른 사실일 수 있다 ('7월 한여름' 을 바꾸면
        # '7월부터 오스트리아 참전' 까지, '8월 19일에 발매' 를 바꾸면 '2006년 8월 19일' 데뷔일까지 바뀐다)
        return "yearless_date_repeated"
    if _is_yearless_date(answer) and any(
        ANNIVERSARY.search(sentence) for sentence in context_sentences(context, pattern)
    ):
        return "anniversary_date"  # '5월 23일 … 이날은 서거 1주기' — 날짜를 옮기면 기념일 설명이 거짓이 된다
    if YEAR_IN_ANSWER.search(answer) and _in_year_range(context, pattern):
        return "year_in_range"  # '취임 1년차(2008년 2월 ~ 2009년 2월)' — 한쪽 끝만 옮기면 기간이 어긋난다
    if not person and _is_computed(answer, context, pattern):
        return "computed_value"  # '가로 8개와 세로 8개, 총 64개' — 다른 수에서 계산되는 값은 혼자 못 바꾼다
    if person and any(
        re.search(pattern.pattern + ORIGINAL_SCRIPT.pattern, chunk.text) for chunk in context
    ):
        return "name_with_original_script"  # 이름 뒤 괄호의 원어 표기(Sotomayor · 足利尊氏)가 원래 이름을 드러낸다
    if person and _person_cue_nearby(context, pattern):
        return (
            "person_cue_nearby"  # '여공 김경숙(당시 21세)' — 후보 이름의 성별 · 나이를 맞출 수 없다
        )
    if any(ERA_YEAR.search(window) for window in windows):
        return "era_year_nearby"  # '메이지 2년' 같은 다른 연도 표기가 원래 해를 드러낸다
    if YEAR_IN_ANSWER.search(answer) and any(
        RELATIVE_YEAR.search(relative_window)
        for relative_window in chain.from_iterable(
            answer_windows(chunk.text, pattern, RELATIVE_WINDOW) for chunk in context
        )
    ):
        return (
            "relative_year_nearby"  # '1985년 시즌 종료 후 … 이듬해 1986년' — 해를 옮기면 어긋난다
        )
    years = QUESTION_YEAR.findall(source.question)
    sentences = list(context_sentences(context, pattern))
    if years and not any(any(year in sentence for sentence in sentences) for year in years):
        # 질문이 묻는 해가 답과 같은 문장에 없다 — '2010년 기준' 질문에 '2019년 현재' 문장을 바꾸게 된다
        return "question_time_mismatch"
    return None


def postcheck(
    answer: str,
    new: str,
    sub_type: str,
    new_context: Sequence[Passage],
    replaced: int,
    *,
    gold_title: str,
) -> str | None:
    """바꾼 뒤 원래 값이 다른 꼴로 남았는지 본다. 남았으면 이유, 깨끗하면 None."""
    joined = squash("".join(chunk.text for chunk in new_context))
    if replaced == 0 or squash(answer) in joined or squash(new) not in joined:
        return "not_clean"  # 원래 표기가 남았거나 바뀐 값이 안 들어갔다
    if sub_type == "person" and any(
        squash(part) in joined for part in answer.split() if len(part) >= NAME_PART_MIN
    ):
        return "name_part_left"  # 이름 일부(성)만 따로 남았다 — '와이어트는 …'
    if sub_type == "date" and any(
        re.search(rf"(?<!\d){year}(?!\d)", joined) for year in YEAR_IN_ANSWER.findall(answer)
    ):
        return "year_left"  # 원래 해가 다른 문장에 따로 남았다
    if sub_type == "person" and replaced > PERSON_REPLACE_MAX:
        return "person_replaced_too_often"  # 문서 주인공급 — 별칭 · 이니셜(녹두장군 · MB)이 남는다
    if sub_type == "person" and _surname_title_left(answer, new_context):
        return "surname_title_left"  # '이 대통령 임기 중에 완성해 성기원 작품으로'
    if sub_type == "number" and _number_left(answer, new_context, gold_title):
        return "number_left"  # '리믹스버전까지 26곡' 인데 '총 23개의 트랙'
    return None


def _surname_title_left(answer: str, context: Sequence[Passage]) -> bool:
    """한국식 이름을 바꾼 문맥에 원래 성 + 직함(이 대통령 · 김 의원)이 남았는지."""
    if name_style(answer) != "ko":
        return False
    titles = "|".join(ROLE_SUFFIXES)
    left = re.compile(rf"(?<![가-힣]){answer[0]}\s*(?:{titles})(?![가-힣])")
    return any(left.search(chunk.text) for chunk in context)


def _number_left(answer: str, context: Sequence[Passage], gold_title: str) -> bool:
    """원래 수(NUMBER_LEFT_MIN 이상)가 정답 문서 청크에 날짜가 아닌 다른 단위로 남았는지 (23곡 → '23개의 트랙')."""
    runs = DIGITS.findall(answer)
    if len(runs) != 1 or int(runs[0]) < NUMBER_LEFT_MIN:
        return False
    left = re.compile(rf"(?<![\d.,]){runs[0]}(?![\d.,])\s*(?!년|월|일|시|분|초|세기)[가-힣]")
    return any(left.search(chunk.text) for chunk in context if chunk.title == gold_title)


def _person_cue_nearby(context: Sequence[Passage], pattern: re.Pattern[str]) -> bool:
    """이름 앞뒤 PERSON_CUE_WINDOW 글자 안에 성별 · 나이 단서(여공 · 당시 21세)가 있는지."""
    return any(
        PERSON_CUE.search(window)
        for window in chain.from_iterable(
            answer_windows(chunk.text, pattern, PERSON_CUE_WINDOW) for chunk in context
        )
    )


def _is_yearless_date(answer: str) -> bool:
    """월 · 일 · 시 · 분만으로 된 날짜 답인지 (8월 19일 · 오전 12시 2분)."""
    kinds = run_kinds(answer, list(DIGITS.finditer(answer)))
    return bool(kinds) and all(kind in BOUNDED for kind in kinds)


def _distinct_spots(context: Sequence[Passage], pattern: re.Pattern[str]) -> list[str]:
    """정답 표기가 나온 서로 다른 자리 (앞뒤 SPOT_WIDTH 글자).

    겹쳐 잘린 청크에 같은 문장이 두 번 나오면 한 자리로 센다 — 잘린 쪽이 다른 쪽에 들어 있다.
    """
    spots: list[str] = []
    for chunk in context:
        for window in answer_windows(chunk.text, pattern, SPOT_WIDTH):
            key = squash(window)
            if not any(key in spot or spot in key for spot in spots):
                spots.append(key)
    return spots


def _in_year_range(context: Sequence[Passage], pattern: re.Pattern[str]) -> bool:
    """정답 연도가 문맥에서 다른 연도와 범위(2008년 2월 ~ 2009년 2월)를 이루는지."""
    for chunk in context:
        text = chunk.text
        for found in pattern.finditer(text):
            if RANGE_AFTER.match(text, found.end()) or RANGE_BEFORE.search(text, 0, found.start()):
                return True
    return False


def _is_computed(answer: str, context: Sequence[Passage], pattern: re.Pattern[str]) -> bool:
    """정답 수가 같은 문장에 나온 다른 수들에서 계산되는지 (_computed_in)."""
    runs = DIGITS.findall(answer)
    if len(runs) != 1:
        return False
    value = int(runs[0])
    return any(
        _computed_in(value, pattern.sub(" ", sentence))
        for sentence in context_sentences(context, pattern)
    )


def _computed_in(value: int, sentence: str) -> bool:
    """정답 자리를 지운 문장에서 value 가 다른 두 수로 계산되는지.

    두 수의 곱 · 합 (가로 8개와 세로 8개, 총 64개 — 연도끼리 더하지 않게 COMPUTED_OPERAND_MAX 미만만),
    덧셈 · 뺄셈 서술이 있으면 크기 상관없이 합 · 차 (2000번대의 속번으로 1000을 더하여 3000번대),
    범위의 개수 (22대부터 30대까지 9회).
    """
    matches = list(DIGITS.finditer(sentence))
    small = [
        int(match.group())
        for match in matches
        if COUNT_MIN <= int(match.group()) < COMPUTED_OPERAND_MAX
    ]
    if any(a * b == value or a + b == value for a, b in combinations(small, 2)):
        return True
    arithmetic = ARITHMETIC.search(sentence) is not None
    for first, second in combinations(matches, 2):  # 나온 순서대로 짝
        a, b = int(first.group()), int(second.group())
        if arithmetic and value in (a + b, abs(a - b)):
            return True
        between = sentence[first.end() : second.start()]
        if len(between) <= RANGE_GAP and RANGE_BETWEEN.search(between) and b - a + 1 == value:
            return True
    return False


def is_person_item(item: SourceItem) -> bool:
    """인물 반사실 대상 · 인물 치환 후보인지: 답이 이름꼴 하나이고 질문이 사람을 묻는다."""
    return (
        item.answer_shape == "person"
        and len(item.answers) == 1
        and is_name(item.answers[0])
        and PERSON_QUESTION.search(item.question) is not None
    )


def _is_counterfactual_shape(item: SourceItem) -> bool:
    """반사실 대상인지: 숫자 · 날짜는 그대로, 인물은 is_person_item 만."""
    return item.answer_shape == "date_number" or is_person_item(item)
