"""답변 생성에 넘길 대화: 프롬프트 파일의 지시는 system 턴, 검색한 문서 · 질문은 user 턴.

문서를 지시와 다른 턴에 두는 것은 간접 인젝션 방어의 일부다 — 문서 안의 문장이 지시처럼 읽히지 않게.
문서 칸은 컨텍스트 조립기(`context_layouts.ask_documents`)가 만든다 — 에이전트 노드와 같은 조립기다.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from playground_llm import Message, ModelSpec, Prompt, PromptLibrary, PromptTemplate, compose
from playground_retrieval.core import Hit

from playground_wiki.generation.context_layouts import (
    AGENT_LAYOUT_VERSION,
    ASK_BUDGET,
    agent_answer_messages,
    ask_documents,
)
from playground_wiki.generation.policy import Profile

PROMPT_DIR = Path(__file__).parent / "prompts"
"""위키 프롬프트 파일 (`{이름}/{버전}.md`). 쓰는 쪽이 시작할 때 `PromptLibrary.load(PROMPT_DIR)` 로 읽는다."""
NOT_FOUND = "주어진 문서에서 답을 찾을 수 없습니다."
"""grounded 프롬프트가 답을 못 찾으면 이 문장만 쓰라고 시킨다. 줄글 답의 거절 판정에 그대로 쓴다."""
SYSTEM_VARIABLES = frozenset({"today", "dump_date", "not_found"})
"""system 조각이 쓸 수 있는 자리표시자. 문서 · 질문은 여기에 들어가지 않는다."""
USER_VARIABLES = frozenset({"context", "question"})
"""user 조각이 정확히 써야 하는 자리표시자."""
CONTEXT_PARAM = "num_ctx"
"""프로필 params 중 모델 문맥 길이 — 문서 칸 예산의 전체 상한으로 쓴다."""


@dataclass(frozen=True)
class ChatPrompt:
    """LLM 에 넘길 대화와 그 프롬프트 버전."""

    messages: tuple[Message, ...]
    """system(지시) → user(문서 · 질문)."""
    version: str
    """조각마다 `이름@버전:해시` 를 `+` 로 이은 것 (system 조각들 → user 조각). 평가 기록의 prompt_version."""


def build_prompt(
    profile: Profile,
    library: PromptLibrary,
    *,
    question: str,
    context: Sequence[Hit],
    today: str,
    dump_date: str,
) -> ChatPrompt:
    """프로필의 프롬프트 조각으로 system · user 턴을 만든다.

    Args:
      profile: 답변 단계의 생성 설정 (system 조각들 · user 조각).
      library: `PROMPT_DIR` 에서 읽은 프롬프트 모음.
      question: 사용자 질문 원문.
      context: 답변 근거로 넘길 검색 결과 (`Decision.context`).
      today: 서버가 아는 오늘 날짜. 사람이 읽는 꼴로 (예: "2026년 10월 24일"). 모델은 오늘을 모른다 —
        값 없이 "덤프 뒤는 모른다"고만 시키면 오늘을 덤프 날짜로 가정하거나 날짜를 지어낸다.
      dump_date: 덤프 기준일. 같은 꼴로 (예: "2026년 10월 1일").

    Raises:
      PromptError: 프로필이 가리키는 프롬프트가 없거나 자리표시자를 채울 수 없을 때.
      BudgetError: 지시 · 문서가 칸 예산(`context_layouts`)에 들어가지 않을 때.
    """
    system = build_system(profile.system, library, today=today, dump_date=dump_date)
    documents = ask_documents(
        context, system=system.text, question=question, budget=context_budget(profile.spec)
    )
    user = compose([library.get(profile.user)], {"context": documents, "question": question})
    return ChatPrompt(
        messages=(Message("system", system.text), Message("user", user.text)),
        version=f"{system.version}+{user.version}",
    )


def build_agent_prompt(
    profile: Profile,
    library: PromptLibrary,
    *,
    question: str,
    context: Sequence[Hit],
    today: str,
    dump_date: str,
) -> ChatPrompt:
    """에이전트 답변 단계의 대화: system 은 단발과 같은 프로필 조각, user 는 `agent_layout` 으로 조립한 질문 · 근거.

    지시(안전 규칙 포함)는 단발과 같은 글이 같은 system 턴에 들어간다. user 턴에는 질문 원문과 모은 근거만 —
    에이전트가 쓴 글(하위 질문 · 검색어 · 메모)은 답변 프롬프트에 넣지 않는다.

    Args:
      profile: 답변 단계의 생성 설정. user 조각은 쓰지 않는다 (칸 배치가 대신한다).
      library: `PROMPT_DIR` 에서 읽은 프롬프트 모음.
      question: 사용자 질문 원문.
      context: 답변 근거로 넘길 검색 결과 (`Decision.context`). 번호는 이 순서 — sources 이벤트와 같다.
      today: 서버가 아는 오늘 날짜 (`build_prompt` 와 같은 꼴).
      dump_date: 덤프 기준일.

    Raises:
      PromptError: 프로필이 가리키는 프롬프트가 없거나 자리표시자를 채울 수 없을 때.
      BudgetError: 지시 · 근거가 칸 예산에 들어가지 않을 때.
    """
    system = build_system(profile.system, library, today=today, dump_date=dump_date)
    messages = agent_answer_messages(
        context, system=system.text, question=question, budget=context_budget(profile.spec)
    )
    return ChatPrompt(messages=messages, version=f"{system.version}+{AGENT_LAYOUT_VERSION}")


def build_system(
    refs: Sequence[str], library: PromptLibrary, *, today: str, dump_date: str
) -> Prompt:
    """지시 조각들(`이름@버전`)을 오늘 · 덤프 날짜 · '못 찾음' 문장으로 채워 이은 system 글을 돌려준다.

    Raises:
      PromptError: 없는 프롬프트를 가리키거나 자리표시자를 채울 수 없을 때.
    """
    parts = [library.get(ref) for ref in refs]
    # 조각마다 쓰는 값이 달라(not_found 는 grounded 만, today 는 base@v3 부터) 쓰는 것만 넘긴다. 빠진 값은 compose 가 잡는다
    values = {"today": today, "dump_date": dump_date, "not_found": NOT_FOUND}
    return compose(parts, _used(parts, values))


def context_budget(spec: ModelSpec) -> int:
    """모델 설정의 문맥 길이(`num_ctx`)를 돌려준다. 없으면 `ASK_BUDGET`."""
    value = spec.params.get(CONTEXT_PARAM)
    return value if isinstance(value, int) and not isinstance(value, bool) else ASK_BUDGET


def korean_date(day: date) -> str:
    """프롬프트 · 고정 답에 넣는 날짜 꼴 ("2026년 10월 24일")."""
    return f"{day.year}년 {day.month}월 {day.day}일"


def format_context(hits: Sequence[Hit]) -> str:
    """검색 결과를 `<document id="1" title="…">` 블록으로 이어 붙인 글을 돌려준다 (`ask_documents` 의 문서 칸만).

    id 는 1부터 hits 순서대로 — `citations.sources(hits)` 와 같은 번호다. 본문 속 블록 태그 흉내는 지운다.
    """
    return ask_documents(hits)


def check_profiles(profiles: Iterable[Profile], library: PromptLibrary) -> None:
    """프로필이 가리키는 프롬프트가 모두 있고, 정해진 자리표시자만 쓰는지 확인한다. 서버를 시작할 때 부른다.

    Raises:
      PromptError: 없는 프롬프트를 가리킬 때.
      ValueError: system 조각이 `SYSTEM_VARIABLES` 밖의 자리표시자를 쓰거나,
        user 조각이 `USER_VARIABLES` 를 정확히 쓰지 않을 때.
    """
    for profile in profiles:
        for ref in profile.system:
            if extra := library.get(ref).variables - SYSTEM_VARIABLES:
                raise ValueError(f"{ref}: system prompt uses unknown placeholders {sorted(extra)}")
        user = library.get(profile.user)
        if user.variables != USER_VARIABLES:
            raise ValueError(
                f"{profile.user}: user prompt must use exactly {sorted(USER_VARIABLES)}, "
                f"got {sorted(user.variables)}"
            )


def _used(parts: Sequence[PromptTemplate], available: Mapping[str, str]) -> dict[str, str]:
    needed = frozenset().union(*(part.variables for part in parts))
    return {name: value for name, value in available.items() if name in needed}
