"""에이전트 그래프 노드: 분해 → 검색 → 판단 → (모자라면 새 검색어로 검색 …) → 답변 근거 고르기.

노드는 얇은 함수다 — LLM 판단은 `Reasoner`, 검색은 단발과 같은 검색 서비스(`Searcher`)를 툴로 부른다.
노드는 시작할 때 `StepStarted` 를 흘리고(받는 쪽이 SSE stage 이벤트로), 바꿀 상태 키만 돌려준다.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langgraph.types import StreamWriter
from playground_retrieval.search import SearchResult

from playground_wiki.agent.evidence import merge_hits, select_answer_hits
from playground_wiki.agent.reasoner import MAX_NEW_QUERIES, Reasoner
from playground_wiki.agent.state import AgentState, Step, StepRecord, StepStarted
from playground_wiki.generation.context_layouts import hit_evidence

MAX_ITERATIONS = 3  # 검색 반복 상한 (docs/plan.md §6.4). 첫 검색(하위 질문) + 새 검색어 두 번


class Searcher(Protocol):
    """질의로 청크를 찾는 툴 — 단발 /ask 와 같은 검색 서비스 (SearchService)."""

    async def search(self, query: str) -> SearchResult:
        """리랭커 점수(그 질의로 잰 것)가 붙은 상위 청크."""
        ...


@dataclass(frozen=True)
class AgentDeps:
    """노드가 쓰는 의존성. `build_graph` 가 노드마다 묶는다."""

    search: Searcher
    """검색 툴 — 하위 질문 · 새 검색어를 그대로 넘기고, 리랭커도 그 검색어로 매긴다 (질의 변형 실험은 채택 없음)."""
    reasoner: Reasoner
    low: float
    """답변 정책의 low — 미만인 근거는 판단 · 답변에 넘기지 않는다."""
    budget: int
    """답변 단계의 전체 토큰 상한 (프로필의 num_ctx) — 답변 근거를 고를 때 쓴다."""
    max_query_chars: int
    """검색어 길이 상한 (리랭커 입력 한도 — /search 와 같은 값). 넘는 하위 질문 · 새 검색어는 자른다."""
    clock: Callable[[], float] = time.perf_counter


async def decompose_node(
    state: AgentState, writer: StreamWriter, *, deps: AgentDeps
) -> dict[str, object]:
    """질문을 하위 질문으로 나눠 첫 검색어로 둔다."""
    writer(StepStarted("decompose"))
    start = deps.clock()
    sub_questions, status = await deps.reasoner.decompose(state["question"], system=state["system"])
    queries = tuple(_clip(text, deps.max_query_chars) for text in sub_questions)
    return {
        "memory": state["memory"].with_sub_questions(sub_questions),
        "pending": queries,
        "decompose": status,
        "steps": _record(state, "decompose", deps, start),
    }


async def search_node(
    state: AgentState, writer: StreamWriter, *, deps: AgentDeps
) -> dict[str, object]:
    """기다리는 검색어를 모두 찾아 근거에 쌓는다. 이미 쓴 검색어는 다시 찾지 않는다.

    리랭커 점수는 그 검색어(하위 질문)로 잰 것이다. 같은 청크는 한 번만 — 더 높은 점수를 남긴다.
    """
    memory = state["memory"]
    queries = [query for query in dict.fromkeys(state["pending"]) if query not in memory.queries]
    writer(StepStarted("search", queries=len(queries)))
    start = deps.clock()
    # 검색 서비스가 GPU 모델을 한 번에 하나씩 쓰도록 안에서 줄 세운다 — 여기서는 순서만 지키며 한꺼번에 넘긴다
    results = await asyncio.gather(*(deps.search.search(query) for query in queries))
    hits = state["hits"]
    for query, result in zip(queries, results, strict=True):
        memory = memory.add_evidence(hit_evidence(result.hits, query=query))
        hits = merge_hits(hits, result.hits)
    return {
        "memory": memory.add_queries(queries).next_iteration(),
        "hits": hits,
        "pending": (),
        "searches": state["searches"] + len(queries),
        "steps": _record(state, "search", deps, start),
    }


async def judge_node(
    state: AgentState, writer: StreamWriter, *, deps: AgentDeps
) -> dict[str, object]:
    """모은 근거로 답할 수 있는지 판단한다. 모자라면 새 검색어를 기다리는 검색어로 둔다."""
    writer(StepStarted("judge"))
    start = deps.clock()
    memory = state["memory"]
    judgment = await deps.reasoner.judge(memory, system=state["system"])
    if judgment.found:
        memory = memory.add_note(judgment.found)
    fresh = [
        _clip(query, deps.max_query_chars)
        for query in judgment.queries
        if query not in memory.queries
    ]
    pending = () if judgment.sufficient else tuple(fresh[:MAX_NEW_QUERIES])
    if judgment.status == "error":
        stop, pending = "judge_error", ()
    elif judgment.sufficient:
        stop = "sufficient"
    elif not pending:
        stop = "no_new_queries"
    else:
        stop = None
    return {
        "memory": memory,
        "pending": pending,
        "judgments": (*state["judgments"], judgment),
        "stop": stop,
        "steps": _record(state, "judge", deps, start),
    }


def compose_node(state: AgentState, *, deps: AgentDeps) -> dict[str, object]:
    """모은 근거에서 답변 근거를 고른다 (low 이상, 칸 예산 안). 반복 상한으로 왔으면 멈춘 이유를 적는다."""
    start = deps.clock()
    chosen = select_answer_hits(state["memory"], state["hits"], low=deps.low, budget=deps.budget)
    return {
        "answer_hits": chosen,
        "stop": state["stop"] or "max_iterations",
        "steps": _record(state, "compose", deps, start),
    }


def route_after_search(state: AgentState) -> Step:
    """검색 다음 노드: 반복 상한에 닿았으면 답변 근거 고르기, 아니면 판단."""
    return "compose" if state["memory"].iterations >= MAX_ITERATIONS else "judge"


def route_after_judge(state: AgentState) -> Step:
    """판단 다음 노드: 새 검색어가 있으면 검색, 없으면(충분 · 새 검색어 없음 · 판단 실패) 답변 근거 고르기."""
    return "search" if state["pending"] else "compose"


def _clip(text: str, limit: int) -> str:
    return text[:limit]


def _record(state: AgentState, step: Step, deps: AgentDeps, start: float) -> tuple[StepRecord, ...]:
    return (*state["steps"], StepRecord(step, round((deps.clock() - start) * 1000, 1)))
