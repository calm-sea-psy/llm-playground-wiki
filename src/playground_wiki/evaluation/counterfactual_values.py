"""반사실 값 고르기 — 인물 · 수 · 날짜 표기를 같은 꼴의 다른 값으로 바꾼다."""

import random
import re
from collections.abc import Sequence, Set
from typing import Any

from playground_evalkit import squash

from playground_wiki.evaluation.counterfactual_checks import keeps_date_order
from playground_wiki.evaluation.counterfactual_rules import (
    BOUNDED,
    CENTURY_MAX,
    CENTURY_SHIFT,
    COUNT_MIN,
    COUNT_MIN_CHANGE,
    DAY_MIN_CHANGE,
    DECADE_WINDOW,
    DIGITS,
    FICTIONAL_NAMES,
    JOSA,
    JOSA_PAIRS,
    LETTERS,
    NEAR_CHOICES,
    NEIGHBOR_WINDOW,
    ORDINAL,
    ROUND_KEEP_ZEROS,
    SPLIT_NUMBER,
    SUBSTITUTE_ATTEMPTS,
    YEAR_MIN_CHANGE,
    YEAR_WINDOW,
    first_group,
    flexible_pattern,
    keeps_order,
    name_style,
    run_kinds,
    run_pattern,
)
from playground_wiki.evaluation.korean import RIEUL_FINAL, final_consonant


def substitute_person(
    answer: str, context_squash: str, rng: random.Random, *, used: Set[str] = frozenset()
) -> str | None:
    """인물명 정답을 같은 표기 계열의 가상 이름(FICTIONAL_NAMES)으로 바꾼다.

    Args:
      answer: 원래 정답 이름.
      context_squash: 공백을 지운 원래 문맥. 여기 나오는 이름은 쓰지 않는다.
      rng: 재현용 난수원.
      used: 이미 다른 문항에 쓴 이름. 다시 쓰지 않는다.

    Returns:
      가상 이름. 표기 계열을 모르거나(name_style 이 None) 남은 이름이 없으면 None.
    """
    style = name_style(answer)
    if style is None:
        return None
    choices = [
        name
        for name in FICTIONAL_NAMES[style]
        if name not in used and squash(name) not in context_squash
    ]
    return rng.choice(choices) if choices else None


def substitute_number(
    value: str,
    texts: Sequence[str],
    rng: random.Random,
    *,
    doc_years: Sequence[int] = (),
    doc_dates: Sequence[tuple[int, int]] = (),
) -> str | None:
    """날짜 · 숫자 답의 숫자를 그럴듯한 다른 값으로 바꾼다. 못 바꾸면 None.

    같은 순서 종류(연도 · 연대 · 세기 · 같은 단위의 수)의 숫자 덩어리는 한 간격으로 함께 옮겨 범위의 앞뒤가
    뒤집히지 않게 하고(19-20세기 → 21-14세기 방지), 문맥에서 답 근처에 나온 같은 종류 값과의 앞뒤 순서를
    지킨다(1987년 … 1989년 … 1990년 이면 1988년). 수는 2 이상, 원래 값의 절반 ~ 1.5배 안에서만 고른다.

    Args:
      value: 원래 정답 문자열. 숫자가 하나 이상 들어 있어야 한다.
      texts: 원래 문맥의 청크 본문들. 답 근처 값과 바뀐 값의 중복을 여기서 본다.
      rng: 재현용 난수원.
      doc_years: 정답 문서 전체의 다른 연도들 (document_years). 연도는 근처 값과 함께 이들과의 순서도 지킨다.
      doc_dates: 연도 없는 월일 답이면 정답 문서 전체의 다른 (월, 일)들 (document_dates). 바뀐 월일은
        이들과의 앞뒤 순서를 지킨다.
    """
    if SPLIT_NUMBER.search(value):
        return None  # 1.25배 같은 소수 · 5·18 같은 이름 속 날짜는 덩어리마다 따로 바뀌어 깨진다
    runs = list(DIGITS.finditer(value))
    if not runs:
        return None
    kinds = run_kinds(value, runs)
    if "count:" in kinds and LETTERS.search(value):
        return None  # 단위 없는 숫자에 글자가 붙은 답은 이름이다 — '뉘르부르크링 2009'
    if "오전" in value or "오후" in value:
        kinds = ["hour12" if kind == "hour" else kind for kind in kinds]  # 오전 13시 방지
    neighbors = _neighbors(value, texts, runs, kinds)
    if neighbors is None:
        return None  # 원래 값이 답 밖 다른 자리에도 있어 바꿔도 남는다
    if "year" in neighbors:
        neighbors["year"] += doc_years
    context_squash = squash("".join(texts))
    for _ in range(SUBSTITUTE_ATTEMPTS):
        new_values = _shift_values(runs, kinds, neighbors, rng)
        if new_values is None:
            return None
        new = _join_runs(value, runs, new_values)
        if (
            new != value
            and squash(new) not in context_squash
            and keeps_date_order(value, new, doc_dates)
        ):
            return new
    return None


def _neighbors(
    value: str, texts: Sequence[str], runs: Sequence[re.Match[str]], kinds: Sequence[str]
) -> dict[str, list[int]] | None:
    """답 근처(NEIGHBOR_WINDOW 글자)에 답과 따로 나온 같은 종류 값들. 바뀐 값은 이들과의 앞뒤 순서를 지킨다.

    서수(제124대)는 근처가 아니라 문맥 전체의 같은 서수를 본다 — 제123대 다이쇼 · 제125대 아키히토가 다른 청크에 있다.

    Returns:
      종류 → 값 목록. 답의 값이 답 밖 근처 자리에도 그대로 있으면 None (바꿔도 원래 값이 남는다).
    """
    own: dict[str, set[int]] = {}
    for run, kind in zip(runs, kinds, strict=True):
        own.setdefault(kind, set()).add(int(run.group()))
    pattern = flexible_pattern(value)
    ordinal = ORDINAL.match(value) is not None
    found: dict[str, list[int]] = {kind: [] for kind in own}
    for text in texts:
        masked = pattern.sub(lambda match: " " * len(match.group()), text)  # 답 자리는 지우고 본다
        windows = (
            [masked]
            if ordinal
            else [
                masked[max(0, match.start() - NEIGHBOR_WINDOW) : match.end() + NEIGHBOR_WINDOW]
                for match in pattern.finditer(text)
            ]
        )
        for window in windows:
            for kind, own_values in own.items():
                if (kind_pattern := run_pattern(kind, ordinal=ordinal)) is None:
                    continue
                values = [
                    int(first_group(found_match)) for found_match in kind_pattern.finditer(window)
                ]
                # 월 · 시는 근처에 같은 값이 흔하다(8월 말) — 그 자리를 그대로 두는 것으로 처리한다 (_shift_values)
                if kind not in BOUNDED and own_values & set(values):
                    return None
                found[kind] += values
    return found


def _shift_values(
    runs: Sequence[re.Match[str]],
    kinds: Sequence[str],
    neighbors: dict[str, list[int]],
    rng: random.Random,
) -> list[int] | None:
    """덩어리마다 새 값. 월 · 일 · 시 · 분은 덩어리마다 범위 안에서 따로, 나머지 종류는 한 간격으로 함께.

    월은 근처 다른 월과의 순서를 지키고, 지킬 수 없으면 그대로 둔다 (8월 19일 → 8월 2x일 — '7월 28일에
    발표 … 9월 말까지 홍보' 사이). 시는 근처에 다른 시각이 있으면 그대로 두고 분만 바꾼다 (사건 순서 유지).
    """
    old = [int(run.group()) for run in runs]
    new = list(old)
    for kind in dict.fromkeys(kinds):  # 나온 순서대로 종류마다 한 번
        indexes = [index for index, run_kind in enumerate(kinds) if run_kind == kind]
        if kind in BOUNDED:
            for index in indexes:
                new[index] = _bounded_value(kind, old[index], neighbors.get(kind, []), rng)
            continue
        values = [old[index] for index in indexes]
        deltas = [
            delta
            for delta in _deltas(kind, values)
            if keeps_order(values, delta, neighbors.get(kind, []))
        ]
        if not deltas:
            return None
        delta = rng.choice(deltas[:NEAR_CHOICES])
        for index in indexes:
            new[index] = old[index] + delta
    return new


def _bounded_value(kind: str, old: int, nearby: Sequence[int], rng: random.Random) -> int:
    """월 · 일 · 시 · 분 덩어리 하나의 새 값. 근처 값과의 순서를 지킬 수 없으면 원래 값 그대로."""
    if kind in ("hour", "hour12") and nearby:
        return old  # 근처에 다른 시각이 있다 — 시를 옮기면 사건 순서가 깨지기 쉽다
    if old in nearby:
        return old  # 같은 달이 근처에 또 나온다 — 그 달은 그대로 둔다
    low, high = BOUNDED[kind]
    min_change = DAY_MIN_CHANGE if kind == "day" else 1
    choices = [
        value
        for value in _near(low, high, old, min_change=min_change)
        if keeps_order([old], value - old, nearby)
    ]
    return rng.choice(choices) if choices else old


def _deltas(kind: str, values: Sequence[int]) -> list[int]:
    """종류에 맞는 간격 후보를 작은 것부터. 모든 덩어리가 옮긴 뒤에도 그 종류에 말이 되는 값이어야 한다."""
    if kind.startswith("count:") and min(values) < COUNT_MIN:
        return []  # 0 · 1 로는 바꾸지 않으니 원래 1 이면 고를 값이 없다 (0개의 파열음 · 1번 반복 방지)
    spans = {"year": YEAR_WINDOW, "decade": DECADE_WINDOW, "century": CENTURY_SHIFT}
    span = spans.get(kind, max(values))
    step = 10 if kind == "decade" else _round_step(values) if kind.startswith("count:") else 1
    ordered = sorted(range(-(span // step) * step, span + 1, step), key=abs)
    return [delta for delta in ordered if delta != 0 and _valid_shifts(kind, values, delta)]


def _round_step(values: Sequence[int]) -> int:
    """딱 떨어지는 수(50 · 1000)는 그 꼴을 지키는 간격으로 — '최초로 1000만' 이 1003만이 되지 않게."""
    zeros = min(len(str(value)) - len(str(value).rstrip("0")) for value in values)
    # 0 이 ROUND_KEEP_ZEROS 개까지는 그 자리 단위로(800 → 700 · 900), 더 많으면 한 자리 아래로(1000 → 1100)
    return 10**zeros if zeros <= ROUND_KEEP_ZEROS else 10 ** (zeros - 1)


def _valid_shifts(kind: str, values: Sequence[int], delta: int) -> bool:
    """모든 덩어리를 delta 만큼 옮긴 값이 그 종류에 말이 되는지."""
    return all(_valid_shift(kind, value, value + delta) for value in values)


def _valid_shift(kind: str, old: int, new: int) -> bool:
    """옮긴 값 new 가 그 종류에 말이 되는지."""
    if kind == "year" and abs(new - old) < YEAR_MIN_CHANGE:
        return False
    if kind in ("year", "decade"):
        return new > 0 and len(str(new)) == len(str(old))  # 자리수를 지킨다 (1976 → 4자리)
    if kind == "century":
        return 1 <= new <= CENTURY_MAX
    # 수 — 2 이상, 원래 값의 절반 ~ 1.5배, 원래 값과 COUNT_MIN_CHANGE 이상 차이 (1445 → 1444 는 오타로 보인다)
    return (
        new >= COUNT_MIN
        and old / 2 <= new <= old * 1.5
        and abs(new - old) >= old * COUNT_MIN_CHANGE
    )


def _join_runs(value: str, runs: Sequence[re.Match[str]], new_values: Sequence[int]) -> str:
    """숫자 덩어리만 새 값으로 갈아 끼운 문자열. 0 으로 시작하던 덩어리는 자리수를 지킨다."""
    pieces, last = [], 0
    for run, number in zip(runs, new_values, strict=True):
        text = run.group()
        pieces.append(value[last : run.start()])
        pieces.append(str(number).zfill(len(text)) if text.startswith("0") else str(number))
        last = run.end()
    pieces.append(value[last:])
    return "".join(pieces)


def _near(
    low: int, high: int, value: int, window: int = NEAR_CHOICES, *, min_change: int = 1
) -> list[int]:
    """[low, high] 에서 value 와 min_change 이상 다른, value 에 가까운 순 앞 window 개. rng 가 이 중에서 고른다."""
    options = sorted(
        (n for n in range(low, high + 1) if abs(n - value) >= min_change),
        key=lambda n: abs(n - value),
    )
    return options[:window]


def replace_in_context(
    context: Sequence[dict[str, Any]], old: str, new: str
) -> tuple[list[dict[str, Any]], int]:
    """문맥의 모든 청크에서 정답 표기를 바꾼 새 문맥과 바꾼 횟수를 돌려준다.

    위키 본문이 띄어쓰기만 다르게 적을 수 있어(정답 선정도 공백 무시로 했다) 글자 사이 공백을 허용해 찾는다.
    바로 뒤 조사(이/가 · 은/는 · 을/를 · 과/와 · 으로/로)는 새 값의 받침에 맞춘다 — '정운찬가' 방지.
    """
    pattern = re.compile(flexible_pattern(old).pattern + JOSA)
    total, out = 0, []
    for chunk in context:
        total += len(pattern.findall(chunk["text"]))
        text = pattern.sub(lambda match: new + _fit_josa(new, match.group("josa")), chunk["text"])
        out.append({**chunk, "text": text})
    return out, total


def _fit_josa(word: str, josa: str | None) -> str:
    """Word 뒤에 올 조사를 받침에 맞춘 꼴로. word 끝이 한글이 아니면 그대로."""
    if josa is None:
        return ""
    final = final_consonant(word[-1]) if word else None
    if final is None:
        return josa
    if josa in ("으로", "로"):
        return "로" if final in (0, RIEUL_FINAL) else "으로"
    with_final, without_final = JOSA_PAIRS[josa]
    return with_final if final else without_final
