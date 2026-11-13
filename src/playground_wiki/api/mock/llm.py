"""목업 LLM: playground_llm.FakeLLM 이 단계 · 노드마다 정해 둔 답을 내고, 조각 사이에 사람이 읽는 만큼의 간격을 둔다."""

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType

from playground_llm import LLM, Generation, GenerationError, Message, ModelSpec, StreamDelta

NOT_FOUND_TAG = "#못찾음"
"""grounded 가 '못 찾음'을 쓴다 — 답을 보내기 전에 partial 로 넘어간다 (mode=agent 면 에이전트로)."""
GENERATION_ERROR_TAG = "#생성오류"
"""답변 단계 생성이 실패한다 — 답 조각을 못 보낸 채 error(generation) 로 끝난다."""
# 조각(단어) 사이 간격 — 화면에서 답이 흘러나오는 모양을 볼 만큼. 시험은 0 으로 끈다
DELAY_SEC = 0.03

# 프롬프트 첫머리로 단계 · 노드를 알아본다 (generation/prompts 의 grounded/v* · agent/decompose · agent/judge)
_GROUNDED_HEADING = "## 답하는 법: 근거 답변"
_DECOMPOSE_HEADING = "## 할 일: 질문 나누기"
_JUDGE_HEADING = "## 할 일: 모은 문서로"

GROUNDED_ANSWER = (
    "찾음\n(목업 답변) 경복궁은 1867년 흥선대원군의 주도로 다시 지어졌습니다 [1]. "
    "왕실의 권위를 세우려는 정책의 하나였습니다 [2]."
)
"""grounded 단계의 답. 첫 줄 표지는 서버가 떼고 보낸다."""
PARTIAL_ANSWER = (
    "### 위키백과 문서에서\n(목업 답변) 문서에는 경복궁이 1867년에 다시 지어졌다는 내용이 있습니다 [1].\n\n"
    "### 문서 밖 일반 지식\n목업 모드라 이 칸은 고정 문구입니다."
)
"""partial 단계의 답."""
SUB_QUESTIONS = ("경복궁 중건", "흥선대원군 정책")
"""질문 나누기 노드의 답 (하위 질문)."""
JUDGMENT: Mapping[str, object] = MappingProxyType(
    {"sufficient": True, "found": "경복궁 중건", "missing": "", "queries": []}
)
"""판단 노드의 답 — 늘 '충분'이라 검색 반복은 한 번이다."""


def mock_reply(spec: ModelSpec, messages: Sequence[Message]) -> str:
    """FakeLLM 의 답 함수: 프롬프트로 단계 · 노드를 알아보고, 질문의 꼬리표대로 고정 답을 낸다.

    Raises:
      GenerationError: 질문에 생성 오류 꼬리표가 있을 때 (답변 단계만 — 에이전트 노드는 실패하지 않는다).
    """
    del spec  # 모델 설정과 상관없이 같은 답
    system, user = messages[0].content, messages[-1].content
    if _DECOMPOSE_HEADING in user:
        return json.dumps({"sub_questions": list(SUB_QUESTIONS)}, ensure_ascii=False)
    if _JUDGE_HEADING in user:
        return json.dumps(dict(JUDGMENT), ensure_ascii=False)
    if GENERATION_ERROR_TAG in user:
        raise GenerationError("mock generation failure (generation error tag)")
    if _GROUNDED_HEADING in system:
        return "못 찾음" if NOT_FOUND_TAG in user else GROUNDED_ANSWER
    return PARTIAL_ANSWER


class PacedLLM:
    """조각 사이에 간격을 두는 LLM — 안의 LLM(FakeLLM)이 한꺼번에 내는 조각을 사람이 읽는 속도로 흘린다.

    스트림은 작업 스레드(`ThreadedStream`)에서 돌아 sleep 이 이벤트 루프를 막지 않는다.

    예)
        llm = PacedLLM(FakeLLM([mock_reply]))
    """

    def __init__(self, llm: LLM, *, delay_sec: float = DELAY_SEC) -> None:
        self._llm = llm
        self._delay_sec = delay_sec

    def generate(
        self,
        spec: ModelSpec,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, object] | None = None,
    ) -> Generation:
        """안의 LLM 의 답을 그대로 돌려준다 (에이전트 노드는 스트림을 쓰지 않는다)."""
        return self._llm.generate(spec, messages, json_schema=json_schema)

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        """안의 LLM 의 조각을 delay_sec 간격으로 흘린다."""
        for item in self._llm.stream(spec, messages):
            if isinstance(item, StreamDelta) and self._delay_sec > 0:
                time.sleep(self._delay_sec)
            yield item
