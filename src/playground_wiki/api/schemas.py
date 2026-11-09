"""/search · /health · /ask 요청 · 응답 형식. /ask 응답은 SSE 라 이벤트 형식은 api/answer.py 에 있다."""

from typing import Literal, Self

from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import SearchResult
from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_CHARS = 300  # 리랭커 입력 = 질의 + 청크(384토큰) ≤ 1024토큰. 한국어 300자는 약 200토큰
MAX_K = 20  # 요청 k 상한 — 리랭커가 매기는 후보 수(retrieval.yaml fusion.top_k)까지만 돌려준다


class SearchRequest(BaseModel):
    """검색 요청."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(
        min_length=1, max_length=MAX_QUERY_CHARS, description="자연어 질문 또는 검색어"
    )
    k: int | None = Field(
        default=None, ge=1, le=MAX_K, description="돌려받을 청크 수. 없으면 서버 기본값(5)"
    )
    doc_ids: list[str] = Field(default=[], description="이 문서 id 들 안에서만 찾는다")
    categories: list[str] = Field(
        default=[], description="이 분류 중 하나에 속한 문서에서만 찾는다"
    )

    def to_filters(self) -> Filters | None:
        """검색 범위. 제한이 없으면 None."""
        if not self.doc_ids and not self.categories:
            return None
        return Filters(doc_ids=frozenset(self.doc_ids), categories=frozenset(self.categories))


class SearchHit(BaseModel):
    """검색된 청크 한 건."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str = Field(description="원문서 제목")
    section: str = Field(description="섹션 제목. 문서 첫머리면 빈 문자열")
    text: str = Field(description='"제목 > 섹션" 머리줄을 포함한 청크 본문')
    score: float = Field(description="리랭커 관련도 0~1. 질의끼리 비교할 수 있다")

    @classmethod
    def from_hit(cls, hit: Hit) -> Self:
        """검색 결과 한 건을 응답 형식으로."""
        chunk = hit.chunk
        return cls(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            score=hit.score,
        )


class SearchResponse(BaseModel):
    """검색 응답."""

    model_config = ConfigDict(frozen=True)

    query: str
    collection: str
    took_ms: float = Field(description="검색 시간(ms). 앞 요청을 기다린 시간은 빼고")
    hits: list[SearchHit] = Field(description="score 내림차순")

    @classmethod
    def from_result(cls, query: str, collection: str, result: SearchResult) -> Self:
        """서비스 결과를 응답 형식으로."""
        return cls(
            query=query,
            collection=collection,
            took_ms=result.took_ms,
            hits=[SearchHit.from_hit(hit) for hit in result.hits],
        )


class HealthResponse(BaseModel):
    """서버 상태. 웹 프론트엔드 사이드바가 쓴다."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    collection: str
    chunks: int = Field(description="BM25 색인에 올린 청크 수")


class AskRequest(BaseModel):
    """답변 요청. 응답은 text/event-stream (stage · sources · delta · done · error)."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, description="자연어 질문")
