"""벤치 세트 · 문턱값 스크립트가 같이 쓰는 검색 돌리기와 검색 설정 표시."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from playground_retrieval.core import Hit

PROGRESS_EVERY = 50  # 진행 줄 간격 (문항) — 수천 건을 몇 분에 돈다


class SearchSettings(Protocol):
    """세트 이름 해시에 넣는 검색 설정 — `SearchConfig` 에서 읽는 칸만."""

    collection: str
    model: str
    candidates: int
    rrf_k: int
    rerank_candidates: int
    reranker_model: str


async def search_all[R](
    search: Callable[[str], Awaitable[R]],
    queries: Sequence[str],
    *,
    on_progress: Callable[[int, int], None],
) -> list[R]:
    """질의를 순서대로 돌린 결과. 서비스가 한 번에 하나씩 돌리므로(GPU 모델 · PG 접속 공유) 차례로 기다린다.

    Args:
      search: 질의 하나를 돌리는 함수. 검색 건수가 다르면 그 값을 묶어서 넘긴다.
      queries: 돌릴 질의.
      on_progress: PROGRESS_EVERY 건마다 (끝낸 수, 전체 수) 로 부른다.
    """
    results = []
    for index, query in enumerate(queries, start=1):
        results.append(await search(query))
        if index % PROGRESS_EVERY == 0:
            on_progress(index, len(queries))
    return results


def search_settings(settings: SearchSettings, *, top_k: int, **extra: object) -> dict[str, object]:
    """세트 meta 와 세트 이름 해시에 넣는 검색 설정 — 키 · 값이 바뀌면 세트 폴더 이름이 바뀐다. extra 는 맨 뒤에 붙는다."""
    return {
        "collection": settings.collection,
        "model": settings.model,
        "candidates": settings.candidates,
        "rrf_k": settings.rrf_k,
        "rerank_candidates": settings.rerank_candidates,
        "reranker_model": settings.reranker_model,
        "top_k": top_k,
        **extra,
    }


def hit_row(hit: Hit) -> dict[str, object]:
    """검색 결과 한 건을 세트 파일의 청크 줄로 — 점수는 6자리로 반올림."""
    return {
        "chunk_id": hit.chunk.id,
        "title": hit.chunk.title,
        "section": hit.chunk.section,
        "text": hit.chunk.text,
        "score": round(hit.score, 6),
    }
