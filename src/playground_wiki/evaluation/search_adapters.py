"""평가 실행이 검색기 자리에 끼우는 어댑터 — 부른 검색을 모으는 쪽과, 부르면 실패하는 closed-book 자리."""

from dataclasses import dataclass

from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation.answer_run import Searcher


@dataclass(frozen=True)
class SearchCall:
    """답변 경로가 부른 검색 한 번."""

    query: str
    titles: tuple[str, ...]
    """찾은 청크의 문서 제목 (리랭커 순서)."""


class RecordingSearch:
    """검색기를 감싸 문항 하나 동안 부른 검색(검색어 · 찾은 문서 제목)과 돌려준 청크를 모은다."""

    def __init__(self, search: Searcher) -> None:
        self._search = search
        self.calls: list[SearchCall] = []
        self.hits: list[Hit] = []
        """돌려준 청크 전부 (검색 차례) — 답변 근거 본문을 기록에 남길 때 출처와 맞춘다."""

    def reset(self) -> None:
        """다음 문항 앞에서 모은 것을 비운다."""
        self.calls = []
        self.hits = []

    async def search(self, query: str) -> SearchResult:
        """감싼 검색기의 결과를 그대로 돌려주고 기록한다."""
        result = await self._search.search(query)
        self.calls.append(SearchCall(query, tuple(hit.chunk.title for hit in result.hits)))
        self.hits.extend(result.hits)
        return result


class NoSearch:
    """closed-book 실행의 검색기 자리 — `AnswerRun` 이 검색기를 받지만 closed-book 은 검색하지 않는다."""

    async def search(self, query: str) -> SearchResult:
        """부르면 실패한다.

        Raises:
          RuntimeError: 늘.
        """
        del query  # 검색기 계약 맞춤. 부르는 것 자체가 잘못이다
        raise RuntimeError("closed-book answers never search")
