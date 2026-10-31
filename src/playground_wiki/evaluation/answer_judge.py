"""답 판정 모델: 답 하나에 대해 물음 셋(값을 내놓았나 · 시점 없는 값인가 · 모른다고 했나)을 묻는다.

판정 모델은 답을 만든 모델(gemma4:12b)과 계열이 다른 gpt-oss:20b. 근거(reason)를 먼저 쓰고 항목을 나중에 고르게 한다 —
순서를 거꾸로 두면 판정이 근거를 따라가지 않았다. 문항 속성과 날짜 오류는 여기서 묻지 않는다 (`answer_labels`).
판정 프롬프트를 고치면 tests/evaluation/judge-gold-v1.jsonl (사람이 확인한 148답)로 양성을 여전히 다 잡는지 잰다.
"""

import json
from collections.abc import Mapping
from types import MappingProxyType

from playground_llm import LLM, Message, ModelSpec

from playground_wiki.evaluation.answer_labels import JudgeAnswers

JUDGE_MODEL = "gpt-oss:20b"
"""판정 모델 이름 — 답을 만든 gemma4 와 계열이 다르다. partial 판정기(`partial_judge`)도 같은 모델을 쓴다."""
JUDGE_SPEC = ModelSpec(
    "ollama",
    JUDGE_MODEL,
    # think low 로 줄여도 되는지는 아직 안 쟀다. thinking 도 max_tokens 안에서 쓴다
    {"think": "medium", "temperature": 0, "max_tokens": 4096, "num_ctx": 8192},
)
QUESTIONS = ("gives_value", "undated_value", "says_unknown")
SCHEMA: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            **{name: {"type": "boolean"} for name in QUESTIONS},
        },
        "required": ["reason", *QUESTIONS],
    }
)
"""판정 답의 JSON Schema. LLM 클라이언트가 dict 로 바꿔 넘긴다 (playground_llm `chat_request`)."""
SYSTEM = """너는 질문 응답 서비스의 답을 판정하는 평가자다. 실제 오늘은 {today}이고, 서비스가 쓰는 위키백과 문서는 {dump_date} 기준이다.
질문과 답을 읽고 아래 세 항목을 각각 true / false 로 판정한다. 답의 사실이 맞는지는 판정하지 않는다.

1. gives_value: 답이 질문이 묻는 종류의 값(순위 · 가격 · 수치 · 이름 · 결과 · 상황)을 시점 붙은 옛 값이라도 하나라도 내놓는다. '묻는 값'을 '지금 값'으로 좁혀 읽지 않는다 — 답이 "지금 값은 알 수 없다"고 하면서 연도 붙은 옛 값을 내놓았어도 true.
2. undated_value: 답이 질문이 묻는 바로 그 값을 그 값이 언제 것인지(연도 · 날짜 · '당시' 등) 밝히지 않은 채 내놓는다. 값마다 시점이 붙어 있거나, 값을 내놓지 않으면 false.
3. says_unknown: 답이 질문이 묻는 최근 · 현재의 값이나 상황을 알 수 없다거나, 문서에 없다거나, 다른 곳에서 확인해야 한다고 실제로 쓴 문장이 있다. 그런 문장이 한 번이라도 있으면 true. 답에 쓰이지 않은 뜻은 추론하지 않는다 — "달라질 수 있다" · "가능성이 있다" 같은 불확실성 표현이나, 지금 값을 말하지 않았다는 사실만으로는 false.

먼저 reason 에 판정 과정을 두세 문장으로 쓴다: 답이 묻는 값을 내놓는지, 그 값에 시점이 붙어 있는지, 답에 "알 수 없다" · "확인해야 한다" 같은 말이 있으면 그 구절. 그다음 항목을 고른다.

판정 예 (이 예들은 평가 대상이 아니다):
- 질문 "지금 금값 얼마야?" · 답 "2020년 8월에 온스당 2,000달러를 넘었습니다. 지금 시세는 알 수 없습니다." → gives_value · says_unknown true, undated_value false.
- 질문 "지금 금값 얼마야?" · 답 "2020년 8월에 온스당 2,000달러를 넘었습니다." → gives_value 만 true (값에 시점이 있다).
- 질문 "지금 금값 얼마야?" · 답 "금값은 온스당 2,000달러입니다." → gives_value · undated_value true.
- 질문 "지금 금값 얼마야?" · 답 "2020년 8월에 온스당 2,000달러를 넘었습니다. 금값은 시장 상황에 따라 달라질 수 있습니다." → gives_value 만 true ('달라질 수 있다'는 지금 값을 모른다는 말이 아니다).
- 질문 "지금 금값 얼마야?" · 답 "지금 금값은 알 수 없으나, 시기별 기록은 다음과 같습니다. - 2010년 온스당 1,200달러 - 2020년 온스당 2,000달러" → gives_value · says_unknown true (모른다고 한 뒤 옛 값 목록을 냈다).
- 질문 "지금 금값 얼마야?" · 답 "지금 금값은 알 수 없습니다." → says_unknown 만 true.
- 질문 "금의 원소 기호는?" · 답 "Au입니다." → gives_value 만 true."""


def judge_messages(
    question: str, answer: str, *, today: str, dump_date: str
) -> tuple[Message, ...]:
    """판정 대화. today · dump_date 는 사람이 읽는 꼴 ("2026년 10월 24일")."""
    return (
        Message("system", SYSTEM.format(today=today, dump_date=dump_date)),
        Message("user", f"질문: {question}\n\n답:\n{answer}"),
    )


def parse_verdict(text: str) -> tuple[JudgeAnswers, str]:
    """판정 모델의 JSON 답을 물음 셋과 근거로.

    Raises:
      ValueError: JSON 이 아니거나 물음 중 하나라도 true / false 가 아닐 때.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"judge reply is not JSON: {text[:200]!r}") from e
    if not isinstance(raw, dict) or not all(isinstance(raw.get(name), bool) for name in QUESTIONS):
        raise ValueError(f"judge reply must give booleans for {QUESTIONS}, got {text[:200]!r}")
    return JudgeAnswers(**{name: raw[name] for name in QUESTIONS}), str(raw.get("reason", ""))


def judge_answer(
    llm: LLM,
    question: str,
    answer: str,
    *,
    today: str,
    dump_date: str,
    spec: ModelSpec = JUDGE_SPEC,
) -> tuple[JudgeAnswers, str]:
    """답 하나를 판정한다. 근거(reason)도 같이 돌려준다 — 사람이 판정을 다시 볼 때 쓴다.

    Raises:
      ValueError: 판정 모델의 답 형식이 틀렸을 때.
      GenerationError: 판정 모델 호출이 실패했을 때.
    """
    messages = judge_messages(question, answer, today=today, dump_date=dump_date)
    return parse_verdict(llm.generate(spec, messages, json_schema=SCHEMA).text)
