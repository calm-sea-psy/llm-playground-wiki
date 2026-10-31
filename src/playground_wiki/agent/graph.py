"""에이전트 그래프 조립: 노드 이름 · 간선은 여기에만.

    START → decompose → search ─┬─(반복 < 상한)→ judge ─┬─(새 검색어)→ search
                                └─(반복 = 상한)→ compose ←┘(충분 · 새 검색어 없음 · 판단 실패)
    compose → END

관문은 그래프 밖(답변 서비스)에서 먼저 본다 — 걸리면 그래프를 돌리지 않는다. 답 생성도 그래프 밖에서 단발과 같은
캐스케이드(grounded → partial)로 흘린다 — 그래프는 답변 근거(`answer_hits`)까지 만든다.
"""

import functools

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from playground_wiki.agent.nodes import (
    AgentDeps,
    compose_node,
    decompose_node,
    judge_node,
    route_after_judge,
    route_after_search,
    search_node,
)
from playground_wiki.agent.state import AgentState

NODE_DECOMPOSE = "decompose"
NODE_SEARCH = "search"
NODE_JUDGE = "judge"
NODE_COMPOSE = "compose"


def build_graph(deps: AgentDeps) -> CompiledStateGraph:
    """노드에 의존성을 묶어 컴파일한 그래프를 돌려준다."""
    graph = StateGraph(AgentState)
    graph.add_node(NODE_DECOMPOSE, functools.partial(decompose_node, deps=deps))
    graph.add_node(NODE_SEARCH, functools.partial(search_node, deps=deps))
    graph.add_node(NODE_JUDGE, functools.partial(judge_node, deps=deps))
    graph.add_node(NODE_COMPOSE, functools.partial(compose_node, deps=deps))
    graph.add_edge(START, NODE_DECOMPOSE)
    graph.add_edge(NODE_DECOMPOSE, NODE_SEARCH)
    graph.add_conditional_edges(
        NODE_SEARCH, route_after_search, {NODE_JUDGE: NODE_JUDGE, NODE_COMPOSE: NODE_COMPOSE}
    )
    graph.add_conditional_edges(
        NODE_JUDGE, route_after_judge, {NODE_SEARCH: NODE_SEARCH, NODE_COMPOSE: NODE_COMPOSE}
    )
    graph.add_edge(NODE_COMPOSE, END)
    return graph.compile()
