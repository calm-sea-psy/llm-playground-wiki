"""단기 메모리(작업 기억): 질문 하나를 처리하는 동안 쌓는 하위 질문 · 검색어 · 근거 · 메모 · 반복 횟수.

저장 · 보관이 없는 불변 상태다 — 갱신은 늘 새 상태를 돌려주고 이전 상태는 그대로 둔다.
에이전트가 만든 글(하위 질문 · 새 검색어 · 메모)은 검색 문서 속 지시를 옮겨 담을 수 있어 data 칸 후보로만 내보낸다.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Self

from playground_wiki.generation.context.slots import Candidate


@dataclass(frozen=True)
class Evidence:
    """검색으로 모은 근거 한 건 (요약하지 않은 원문)."""

    key: str
    """중복 판단 키 (예: 청크 id)."""
    text: str
    score: float
    """선별 점수 (예: 원문 질문으로 잰 리랭커 점수)."""
    query: str
    """이 근거를 찾은 검색어."""
    group: str | None = None
    """묶음 (예: 문서 id)."""
    # 해시에서는 뺀다 (매핑은 해시가 안 된다). 같은지 비교에는 들어간다
    attributes: Mapping[str, str] = field(default_factory=dict, hash=False)
    """항목 태그에 붙일 속성 (예: `{"title": "경복궁"}`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def as_candidate(self, slot: str) -> Candidate:
        """`slot` 칸에 넣을 후보로 바꾼 것을 돌려준다."""
        return Candidate(
            slot=slot,
            text=self.text,
            score=self.score,
            key=self.key,
            group=self.group,
            attributes=self.attributes,
        )


@dataclass(frozen=True)
class WorkingMemory:
    """질문 하나의 작업 기억.

    예)
        memory = WorkingMemory("세종대왕의 아버지가 세운 궁궐은?")
        memory = memory.with_sub_questions(["세종대왕의 아버지는?", "태종이 세운 궁궐은?"])
        memory = memory.add_evidence(found).next_iteration()
    """

    question: str
    """사용자 질문 원문."""
    sub_questions: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    """지금까지 쓴 검색어 (새 검색어 포함), 쓴 순서대로 중복 없이."""
    evidence: tuple[Evidence, ...] = ()
    """모은 근거, 처음 찾은 순서대로 — 뒤에 붙여 나가 노드 프롬프트의 앞부분이 그대로다. key 는 하나씩."""
    notes: tuple[str, ...] = ()
    """지난 반복에서 찾은 사실 메모. 출처 번호와 함께 한 줄씩 (에이전트 자신의 기록만 요약한다)."""
    iterations: int = 0
    """끝낸 검색 반복 횟수."""

    def __post_init__(self) -> None:
        # 목록을 넘겨도 밖에서 고친 것이 상태에 번지지 않게
        object.__setattr__(self, "sub_questions", tuple(self.sub_questions))
        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations=}")

    def with_sub_questions(self, sub_questions: Sequence[str]) -> Self:
        """하위 질문을 바꿔 넣은 새 상태를 돌려준다 (분해 노드의 결과)."""
        return replace(self, sub_questions=tuple(sub_questions))

    def add_queries(self, queries: Iterable[str]) -> Self:
        """이미 쓴 것을 뺀 검색어를 뒤에 붙인 새 상태를 돌려준다."""
        added = list(self.queries)
        for query in queries:
            if query not in added:
                added.append(query)
        return replace(self, queries=tuple(added))

    def add_evidence(self, items: Iterable[Evidence]) -> Self:
        """근거를 뒤에 붙인 새 상태를 돌려준다.

        이미 있는 key 는 점수가 더 높을 때만 바꾸고, 자리는 처음 찾은 자리를 지킨다.
        """
        merged = list(self.evidence)
        index = {item.key: i for i, item in enumerate(merged)}
        for item in items:
            seen = index.get(item.key)
            if seen is None:
                index[item.key] = len(merged)
                merged.append(item)
            elif item.score > merged[seen].score:
                merged[seen] = item
        return replace(self, evidence=tuple(merged))

    def add_note(self, note: str) -> Self:
        """찾은 사실 메모 한 줄을 뒤에 붙인 새 상태를 돌려준다."""
        return replace(self, notes=(*self.notes, note))

    def next_iteration(self) -> Self:
        """반복 횟수를 하나 올린 새 상태를 돌려준다."""
        return replace(self, iterations=self.iterations + 1)


def memory_candidates(
    memory: WorkingMemory,
    *,
    question: str | None = None,
    sub_questions: str | None = None,
    queries: str | None = None,
    evidence: str | None = None,
    notes: str | None = None,
) -> list[Candidate]:
    """작업 기억을 data 칸 후보로 바꾼 것을 돌려준다. 칸 이름이 None 인 항목은 넣지 않는다.

    Args:
      memory: 바꿀 작업 기억.
      question: 질문 원문을 넣을 칸.
      sub_questions: 하위 질문을 넣을 칸.
      queries: 쓴 검색어를 넣을 칸.
      evidence: 모은 근거를 넣을 칸. 점수는 근거의 점수 그대로.
      notes: 사실 메모를 넣을 칸.

    Returns:
      칸마다 기억에 쌓인 순서대로. 근거 밖의 항목은 점수 0 — 예산이 모자라면 뒤에서부터 빠진다.
    """
    candidates: list[Candidate] = []
    if question is not None:
        candidates.append(Candidate(slot=question, text=memory.question))
    texts = ((sub_questions, memory.sub_questions), (queries, memory.queries))
    for slot, values in texts:
        if slot is not None:
            candidates.extend(Candidate(slot=slot, text=text, key=text) for text in values)
    if evidence is not None:
        candidates.extend(item.as_candidate(evidence) for item in memory.evidence)
    if notes is not None:
        candidates.extend(Candidate(slot=notes, text=note) for note in memory.notes)
    return candidates
