import json
from datetime import date
from pathlib import Path

import pytest
from playground_llm import FakeLLM

from playground_wiki.evaluation import (
    JudgeAnswers,
    Traits,
    date_error,
    judge_answer,
    label_answer,
    parse_verdict,
)

TODAY, DUMP = date(2026, 10, 24), date(2026, 10, 1)
# 사람이 확인한 148답 (2026-10-24, 프롬프트 v2 · v3 변형) + 판정 모델이 놓쳤던 꼴 1 — 판정기의 회귀 세트. 판정 모델 회귀는
# scripts/judge_check.py 로 (LLM 이 필요하다). 날짜 오류는 규칙이라 여기서 바로 잰다
GOLD = Path(__file__).parent / "judge-gold-v1.jsonl"


def _gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line]


def test_date_rule_matches_every_human_label():
    rows = _gold()
    wrong = [
        (r["id"], r["variant"])
        for r in rows
        if date_error(r["answer"], today=TODAY, dump_date=DUMP) != r["gold"]["date_error"]
    ]

    assert len(rows) == 149
    assert sum(r["gold"]["date_error"] for r in rows) == 4
    assert wrong == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("현재 날짜가 2026년 10월 1일 기준이므로 어제가 언제인지 알 수 없습니다.", True),
        ("현재 시점(2026년 10월 1일 기준)의 수치는 문서에 없습니다.", True),
        (
            "현재 날짜가 2025년 1월 26일인지 확인이 불가능하므로",
            True,
        ),  # 덤프 날짜가 아니면 의문형도 오류
        (
            "현재 시점이 2026년 10월 1일 이전인지 이후인지 알 수 없습니다.",
            False,
        ),  # 덤프 날짜의 의문형은 맞는 추론
        ("오늘 날짜는 2026년 10월 24일이므로 어제는 23일입니다.", False),
        ("현재 시점(2026년 10월)의 매매가는 문서에 없습니다.", False),  # 오늘의 달
        ("2013년 1월 발표에 따르면 늘고 있었습니다.", False),  # 값에 붙은 옛 날짜는 오늘이 아니다
    ],
)
def test_date_rule(text: str, expected: bool):
    assert date_error(text, today=TODAY, dump_date=DUMP) is expected


REALTIME = Traits(should_say_unknown=True, answer_in_context=False)
ANSWERABLE = Traits(should_say_unknown=False, answer_in_context=True)
NOT_IN_DOCS = Traits(should_say_unknown=False, answer_in_context=False)


@pytest.mark.parametrize(
    ("traits", "judge", "expected"),
    [
        # 실시간 질문: 날짜 붙은 옛 값만 → 미표시, 시점 없는 값 → 옛 값을 지금 값처럼까지
        (
            REALTIME,
            JudgeAnswers(True, False, False),
            {"stale_as_current": False, "no_current_disclaimer": True},
        ),
        (
            REALTIME,
            JudgeAnswers(True, True, False),
            {"stale_as_current": True, "no_current_disclaimer": True},
        ),
        (
            REALTIME,
            JudgeAnswers(True, True, True),
            {"stale_as_current": False, "no_current_disclaimer": False},
        ),
        # 문서에 답이 있는 질문: 모른다고만 → 과잉 거절, 값과 함께 모른다 → 불필요한 모름
        (
            ANSWERABLE,
            JudgeAnswers(False, False, True),
            {"over_refusal": True, "extra_caution": False},
        ),
        (
            ANSWERABLE,
            JudgeAnswers(True, False, True),
            {"over_refusal": False, "extra_caution": True},
        ),
        (
            ANSWERABLE,
            JudgeAnswers(True, False, False),
            {"over_refusal": False, "extra_caution": False},
        ),
        # 문서에 답이 없으면 모른다고 해도 과잉 거절이 아니다 (분모에서 뺀다)
        (
            NOT_IN_DOCS,
            JudgeAnswers(False, False, True),
            {"over_refusal": None, "extra_caution": False},
        ),
    ],
)
def test_labels_are_computed_from_traits_and_judge(
    traits: Traits, judge: JudgeAnswers, expected: dict
):
    labels = label_answer("답", traits, judge, today=TODAY, dump_date=DUMP)

    assert {name: getattr(labels, name) for name in expected} == expected


def test_realtime_labels_do_not_apply_to_answerable_questions():
    labels = label_answer(
        "답", ANSWERABLE, JudgeAnswers(True, True, False), today=TODAY, dump_date=DUMP
    )

    assert labels.stale_as_current is None
    assert labels.no_current_disclaimer is None


def test_judge_parses_reason_first_json():
    reply = '{"reason": "값에 시점이 있다", "gives_value": true, "undated_value": false, "says_unknown": false}'
    llm = FakeLLM([reply])

    verdict, reason = judge_answer(llm, "지금 금값?", "2020년 2,000달러", today="t", dump_date="d")

    assert verdict == JudgeAnswers(gives_value=True, undated_value=False, says_unknown=False)
    assert reason == "값에 시점이 있다"
    assert llm.calls[0].messages[0].role == "system"


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '{"gives_value": true}',
        '{"gives_value": "yes", "undated_value": false, "says_unknown": false}',
    ],
)
def test_judge_rejects_malformed_replies(reply: str):
    with pytest.raises(ValueError):
        parse_verdict(reply)
