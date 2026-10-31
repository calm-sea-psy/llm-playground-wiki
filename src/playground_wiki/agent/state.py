"""에이전트 그래프 상태 (LangGraph 상태 — 경계 모듈)와 노드가 흘리는 진행 알림.

단기 메모리(질문 하나를 처리하는 동안의 작업 기억)는 여기 `memory` 에 둔다 — 저장 · 보관이 없어 별도 모듈은 과하다
(context-design §8). 상태 값은 모두 바꿀 수 없는 값(튜플 · frozen dataclass)이라 노드는 새 값을 돌려준다.
"""

from dataclasses import dataclass
from typing import Literal, TypedDict, get_args

from playground_retrieval.core import Hit

from playground_wiki.generation.context import WorkingMemory

Step = Literal["decompose", "search", "judge", "compose"]
"""그래프 노드 (거친 차례대로 진단의 steps 에 남는다)."""
STEPS: tuple[str, ...] = get_args(Step)
StopReason = Literal["sufficient", "max_iterations", "no_new_queries", "judge_error"]
"""검색 반복을 멈춘 이유. judge_error 는 판단 노드가 실패해 모은 근거로 답한 것."""
STOP_REASONS: tuple[str, ...] = get_args(StopReason)
DecomposeStatus = Literal["ok", "fallback", "error"]
"""질문 나누기 결과. fallback 은 하위 질문이 하나도 안 나와 원문 하나로, error 는 LLM 실패로 원문 하나로 간 것."""
JudgeStatus = Literal["ok", "error"]


@dataclass(frozen=True)
class StepStarted:
    """노드가 일을 시작했다는 알림 (LangGraph custom 스트림). 받는 쪽이 SSE stage 이벤트로 바꾼다."""

    step: Step
    queries: int = 0
    """search 노드면 이번에 찾을 검색어 수."""


@dataclass(frozen=True)
class StepRecord:
    """끝난 노드 하나의 기록 (진단용)."""

    step: Step
    ms: float
    """노드가 걸린 시간 (ms)."""


@dataclass(frozen=True)
class Judgment:
    """근거 판단 노드의 결과."""

    sufficient: bool
    found: str
    """지금까지 문서에서 찾은 사실 한 줄 (에이전트가 쓴 글 — 다음 노드에는 태그 안 data 로만)."""
    missing: str
    """빠진 사실 한 줄."""
    queries: tuple[str, ...]
    """새 검색어 (빈 글 · 겹친 것을 뺀 것)."""
    status: JudgeStatus


class AgentState(TypedDict):
    """그래프 상태. 노드는 바꿀 키만 돌려준다."""

    question: str
    """사용자 질문 원문."""
    system: str
    """노드 공통 고정 지시 (base 프롬프트 — 요청마다 오늘 날짜가 들어간다)."""
    memory: WorkingMemory
    """하위 질문 · 쓴 검색어 · 모은 근거 · 메모 · 끝낸 검색 반복 수."""
    hits: tuple[Hit, ...]
    """모은 청크 — 청크 id 하나씩, 높은 점수, 처음 찾은 순서. 근거(memory.evidence)와 같은 청크다."""
    pending: tuple[str, ...]
    """다음 검색 반복에서 찾을 검색어."""
    searches: int
    """검색 서비스를 부른 횟수."""
    decompose: DecomposeStatus | None
    judgments: tuple[Judgment, ...]
    stop: StopReason | None
    steps: tuple[StepRecord, ...]
    answer_hits: tuple[Hit, ...]
    """compose 노드가 고른 답변 근거 (low 이상, 칸 예산 안). 비었으면 답변 정책이 refuse 로 보낸다."""


def initial_state(question: str, *, system: str) -> AgentState:
    """질문 하나를 시작하는 상태를 돌려준다."""
    return AgentState(
        question=question,
        system=system,
        memory=WorkingMemory(question),
        hits=(),
        pending=(),
        searches=0,
        decompose=None,
        judgments=(),
        stop=None,
        steps=(),
        answer_hits=(),
    )
