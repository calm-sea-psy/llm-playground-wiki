"""라우터가 Depends 로 받는 의존성. lifespan 이 app.state 에 둔 객체를 꺼내 준다."""

from typing import Annotated

from fastapi import Depends, Request
from playground_retrieval.search import SearchService
from playground_retrieval.store.qdrant import QdrantMode

from playground_wiki.api.answer import AnswerService


def get_search_service(request: Request) -> SearchService:
    """앱이 시작할 때 lifespan 이 만든 검색 서비스."""
    return request.app.state.search_service


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]


def get_answer_service(request: Request) -> AnswerService:
    """앱이 시작할 때 lifespan 이 만든 답변 서비스 (관문 · 검색 · 정책 · LLM 을 잇는다)."""
    return request.app.state.answer_service


AnswerServiceDep = Annotated[AnswerService, Depends(get_answer_service)]


def get_qdrant_mode(request: Request) -> QdrantMode | None:
    """검색이 붙은 Qdrant 의 위치. 목업(RAG_IMPL=mock)이면 None."""
    return request.app.state.qdrant_mode


QdrantModeDep = Annotated[QdrantMode | None, Depends(get_qdrant_mode)]
