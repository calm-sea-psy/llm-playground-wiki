"""목업 검색기: 예시 검색 응답(contracts/examples/search_response.json)의 청크를 돌려준다. 질문의 꼬리표로 점수 구간을 고른다."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from playground_corpus import Chunk
from playground_retrieval.core import Filters, Hit

from playground_wiki.api.schemas import SearchHit, SearchResponse
from playground_wiki.generation import Thresholds

PARTIAL_TAG = "#일반"
"""1위 점수를 partial 구간(low ≤ 점수 < high)으로 내린다 — 문서 내용과 일반 지식을 나눠 답한다."""
REFUSE_TAG = "#거절"
"""검색 결과를 비운다 — refuse 단계의 고정 문구로 끝난다."""
SEARCH_ERROR_TAG = "#검색오류"
"""검색이 실패한다 — /ask 는 error(internal), /search 는 500."""
# 줄인 점수의 소수 자리 — 예시 점수(둘째 자리)는 그대로 살고 곱셈의 부동소수 꼬리만 잘린다
SCORE_DIGITS = 6


@dataclass(frozen=True)
class MockRetriever:
    """예시 청크를 점수 내림차순으로 돌려주는 검색기 (playground_retrieval Retriever 계약).

    예)
        retriever = MockRetriever.from_example(Path("contracts/examples/search_response.json"), thresholds)
        retriever.search("경복궁을 중건한 사람은? #일반", k=5)  # 1위 점수가 partial 구간
    """

    hits: tuple[Hit, ...]
    """예시의 청크 · 점수 (점수 내림차순). 1위 점수는 high 이상이다."""
    collection: str
    thresholds: Thresholds

    @classmethod
    def from_example(cls, path: Path, thresholds: Thresholds) -> Self:
        """예시 검색 응답 파일로 만든다. 청크 순번은 chunk_id 의 끝 번호, 토큰 수는 글자 수로 어림한다.

        Raises:
          ValueError: 예시에 청크가 없거나 1위 점수가 high 미만일 때 (꼬리표 없는 질문이 grounded 로 가지 않는다).
        """
        example = SearchResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))
        hits = tuple(
            sorted(
                (Hit(chunk=_chunk(hit), score=hit.score) for hit in example.hits),
                key=lambda hit: hit.score,
                reverse=True,
            )
        )
        if not hits or hits[0].score < thresholds.high:
            raise ValueError(f"{path}: top score must be >= thresholds.high {thresholds.high}")
        return cls(hits=hits, collection=example.collection, thresholds=thresholds)

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """질의의 꼬리표대로 예시 청크 k건을 돌려준다. 필터는 doc_ids 만 본다 (예시에 분류가 없다).

        Raises:
          RuntimeError: 질의에 검색 오류 꼬리표가 있을 때.
        """
        if SEARCH_ERROR_TAG in query:
            raise RuntimeError("mock search failure (search error tag)")
        if REFUSE_TAG in query:
            return []
        hits = [hit for hit in self.hits if not filters or _allowed(hit, filters)]
        if PARTIAL_TAG in query:
            hits = self._partial(hits)
        return hits[:k]

    def _partial(self, hits: list[Hit]) -> list[Hit]:
        """1위 점수가 low 와 high 의 한가운데가 되게 점수를 같은 비율로 줄인 결과."""
        if not hits:
            return hits
        scale = (self.thresholds.low + self.thresholds.high) / 2 / hits[0].score
        return [Hit(chunk=hit.chunk, score=round(hit.score * scale, SCORE_DIGITS)) for hit in hits]


def _chunk(hit: SearchHit) -> Chunk:
    return Chunk(
        id=hit.chunk_id,
        doc_id=hit.doc_id,
        title=hit.title,
        section=hit.section,
        text=hit.text,
        seq=int(hit.chunk_id.rsplit("-", 1)[-1]),
        token_count=len(hit.text),
    )


def _allowed(hit: Hit, filters: Filters) -> bool:
    return not filters.doc_ids or hit.chunk.doc_id in filters.doc_ids
