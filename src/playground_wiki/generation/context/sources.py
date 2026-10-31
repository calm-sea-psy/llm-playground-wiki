"""후보 출처 약속: 검색 · 도구 결과 · 기억 조회가 모두 같은 모양으로 후보를 내준다 (수집 단계)."""

import asyncio
from collections.abc import Sequence
from typing import Protocol

from playground_wiki.generation.context.slots import Candidate


class CandidateSource(Protocol):
    """질문 · 예산을 받아 칸 · 점수가 붙은 후보를 돌려주는 출처.

    장기 메모리 조회도 이 약속을 따른다 — 기억이 없는 프로젝트는 빈 목록을 돌려주는 구현을 끼운다.
    """

    async def collect(self, query: str, budget: int) -> Sequence[Candidate]:
        """`query` 와 관련된 후보를 돌려준다. budget 은 토큰 어림 — 넘겨 줘도 조립기가 다시 고른다."""
        ...


async def collect_candidates(
    sources: Sequence[CandidateSource], query: str, budget: int
) -> list[Candidate]:
    """출처들을 동시에 불러 후보를 출처 순서대로 이어 붙인 것을 돌려준다.

    출처 하나가 실패하면 그 예외가 그대로 올라온다 — 빠진 근거로 답하지 않게.
    """
    results = await asyncio.gather(*(source.collect(query, budget) for source in sources))
    candidates: list[Candidate] = []
    for result in results:
        candidates.extend(result)
    return candidates
