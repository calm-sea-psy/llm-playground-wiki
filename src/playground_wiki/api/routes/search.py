"""/search · /health 라우터."""

from fastapi import APIRouter

from playground_wiki.api.deps import QdrantModeDep, SearchServiceDep
from playground_wiki.api.schemas import HealthResponse, SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search")
async def search(request: SearchRequest, service: SearchServiceDep) -> SearchResponse:
    """질문과 관련 높은 위키백과 청크를 Hybrid(Dense + BM25) → 리랭커 순서로 찾는다."""
    result = await service.search(request.query, request.k, request.to_filters())
    return SearchResponse.from_result(request.query, service.collection, result)


@router.get("/health")
async def health(service: SearchServiceDep, qdrant: QdrantModeDep) -> HealthResponse:
    """서버가 검색할 준비가 됐는지와 검색 대상 컬렉션 · Qdrant 위치."""
    return HealthResponse(
        status="ok", collection=service.collection, chunks=service.chunk_count, qdrant=qdrant
    )
