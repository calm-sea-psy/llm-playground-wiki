"""모은 근거 다루기: 청크 합치기 · 칸 후보로 바꾸기 · 답변 근거 고르기.

여러 검색어가 근거를 모으면 점수만으로 고를 때 한 검색어(대개 원문에 가까운 첫 하위 질문)의 청크가 칸을 다 차지한다.
그래서 고를 때는 검색어마다 순위를 돌아가며 — 모든 검색어의 1위 → 2위 … 차례로, 같은 순위면 리랭커 점수로 — 뽑는다.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from playground_retrieval.core import Hit

from playground_wiki.generation.context import (
    Candidate,
    CharTokenCounter,
    ContextAssembler,
    WorkingMemory,
)
from playground_wiki.generation.context_layouts import DOCUMENTS, agent_layout

# 고르는 점수 = −(검색어 안 순위) + 리랭커 점수 × 이 값. 점수(0~1)에 곱해 순위 한 칸(1)보다 작게 — 순위가 먼저다
_SCORE_WEIGHT = 0.5


def merge_hits(found: Sequence[Hit], new: Iterable[Hit]) -> tuple[Hit, ...]:
    """이미 찾은 청크(found) 뒤에 new 를 붙인 목록을 돌려준다 — 청크 id 하나씩, 더 높은 점수를 남기고 자리는 처음 찾은 자리."""
    merged = list(found)
    index = {hit.chunk.id: i for i, hit in enumerate(merged)}
    for hit in new:
        seen = index.get(hit.chunk.id)
        if seen is None:
            index[hit.chunk.id] = len(merged)
            merged.append(hit)
        elif hit.score > merged[seen].score:
            merged[seen] = hit
    return tuple(merged)


def ranked_candidates(memory: WorkingMemory, *, slot: str, low: float) -> list[Candidate]:
    """근거를 `slot` 칸 후보로 바꾼 것을 돌려준다 — low 미만은 빼고, 근거를 찾은 순서 그대로.

    후보의 점수는 고르는 점수(검색어 안 순위가 먼저, 같은 순위면 리랭커 점수)다 — 칸 예산이 모자라면 낮은 순위부터 빠진다.
    """
    kept = [item for item in memory.evidence if item.score >= low]
    by_query: dict[str, list[float]] = defaultdict(list)
    for item in kept:
        by_query[item.query].append(item.score)
    for scores in by_query.values():
        scores.sort(reverse=True)
    candidates = []
    for item in kept:
        rank = by_query[item.query].index(item.score) + 1
        candidate = item.as_candidate(slot)
        candidates.append(
            Candidate(
                slot=candidate.slot,
                text=candidate.text,
                score=-rank + item.score * _SCORE_WEIGHT,
                key=candidate.key,
                group=candidate.group,
                attributes=candidate.attributes,
            )
        )
    return candidates


def select_answer_hits(
    memory: WorkingMemory, hits: Sequence[Hit], *, low: float, budget: int
) -> tuple[Hit, ...]:
    """답변 근거로 넘길 청크를 돌려준다 — low 이상, 에이전트 칸 배치의 근거 칸 예산 안, 검색어마다 순위를 돌아가며.

    Args:
      memory: 모은 근거 (key 는 청크 id).
      hits: 같은 근거의 검색 결과 (`merge_hits`).
      low: 답변 정책의 low — 미만은 근거로 넘기지 않는다.
      budget: 전체 토큰 상한 (답변 단계 프로필의 `num_ctx`). 근거 칸 예산이 먼저 닿는다.

    Returns:
      근거를 찾은 순서대로. 남은 근거가 없으면 빈 튜플 (답변 정책이 refuse 로 보낸다).
    """
    candidates = ranked_candidates(memory, slot=DOCUMENTS, low=low)
    context = ContextAssembler(agent_layout(budget), CharTokenCounter()).assemble({}, candidates)
    by_id = {hit.chunk.id: hit for hit in hits}
    return tuple(by_id[str(placed.candidate.key)] for placed in context.slot(DOCUMENTS).placed)
