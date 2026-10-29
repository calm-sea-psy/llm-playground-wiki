"""F4 벤치 문항을 실제 답변 경로(줄글)로 다시 잴 때의 입력: 세트 문항과, 문항에 저장된 상위 5 청크를 돌려주는 검색기.

검색을 다시 돌리지 않는다 — 반사실 · 간접 인젝션 청크는 고친 것이라 색인에 없고, 문서 없음은 다시 검색하면 정답 문서가
돌아온다. 저장된 청크와 리랭커 점수를 그대로 검색 결과로 넘기면 답변 정책이 그 점수로 단계를 고른다 (고친 청크의 점수는
고치기 전 점수다). F4 JSON 벤치와 입력이 같아 문항마다 짝지어 비교할 수 있다.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast, get_args

from playground_corpus import Chunk
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation.answer_run import ContextChunk, EvalItem

F4Kind = Literal[
    "answerable",
    "no_doc",
    "no_fact",
    "counterfactual",
    "benign_sensitive",
    "injection_indirect",
    "injection_direct",
]
"""줄글로 다시 재는 F4 문항 종류 (세트 빌더가 붙인 kind). answerable 은 과잉 거절의 대조군."""
F4_KINDS: tuple[str, ...] = get_args(F4Kind)
INJECTION_KINDS = frozenset({"injection_indirect", "injection_direct"})
"""방어 문구 비교 변형을 돌리는 종류."""


@dataclass(frozen=True)
class F4Item:
    """F4 세트 문항 하나 (items.jsonl 한 줄) — 줄글 채점에 쓰는 필드만."""

    id: str
    set_name: str
    """문항이 든 세트 폴더 이름 (LLM 레포 bench/data/ 아래)."""
    kind: F4Kind
    question: str
    """모델에 넘기는 질문. 직접 인젝션은 원래 질문 뒤에 공격 문장이 붙어 있다."""
    answers: tuple[str, ...]
    """정답 표기 전부. 반사실은 바꾼 값(문맥을 따른 답), 답 없음 종류는 문맥에 없는 참값."""
    gold_title: str
    context: tuple[ContextChunk, ...]
    """세트에 저장된 상위 5 청크 — 리랭커 점수 내림차순. 반사실 · 간접 인젝션은 고친 본문에 고치기 전 점수."""
    original_answers: tuple[str, ...] = ()
    """반사실의 원래 값. 다른 종류면 비어 있다."""
    canary: str | None = None
    """인젝션의 표식 문자열 — 답에 나오면 공격 성공. 다른 종류면 None."""

    def __post_init__(self) -> None:
        if self.kind not in F4_KINDS:
            raise ValueError(f"unknown F4 kind {self.kind!r} for {self.id=}")
        if not self.answers or not self.context:
            raise ValueError(f"F4 item needs answers and context, got {self.id=}")
        if self.kind == "counterfactual" and not self.original_answers:
            raise ValueError(f"counterfactual item needs original_answers, got {self.id=}")
        if self.kind in INJECTION_KINDS and not self.canary:
            raise ValueError(f"injection item needs a canary, got {self.id=}")

    @classmethod
    def from_json(cls, row: Mapping[str, Any], set_name: str) -> Self:
        """세트 items.jsonl 한 줄에서. 줄글 채점에 쓰지 않는 키(strata · injection · attack …)는 무시한다."""
        return cls(
            id=str(row["id"]),
            set_name=set_name,
            kind=cast("F4Kind", row["kind"]),
            question=str(row["question"]),
            answers=tuple(str(answer) for answer in row["answers"]),
            gold_title=str(row["gold_title"]),
            context=tuple(
                ContextChunk(
                    chunk_id=str(chunk["chunk_id"]),
                    title=str(chunk["title"]),
                    section=str(chunk["section"]),
                    text=str(chunk["text"]),
                    score=float(chunk["score"]),
                )
                for chunk in sorted(row["context"], key=lambda chunk: chunk["rank"])
            ),
            original_answers=tuple(str(answer) for answer in row.get("original_answers", ())),
            canary=row.get("canary"),
        )

    def eval_item(self) -> EvalItem:
        """답변 경로 실행 기록에 실을 문항 (source f4, category 에 kind)."""
        return EvalItem(
            id=self.id,
            source="f4",
            question=self.question,
            gold_title=self.gold_title,
            gold_answer=self.answers[0],
            category=self.kind,
        )

    def search_result(self) -> SearchResult:
        """저장된 청크를 검색 결과로 — 답변 경로가 이 점수로 단계를 고르고 이 본문을 문맥으로 넘긴다."""
        return SearchResult(hits=tuple(_hit(chunk) for chunk in self.context), took_ms=0.0)


@dataclass(frozen=True)
class F4Set:
    """읽어 들인 세트 하나 — 실행 meta 에 남겨 세트가 바뀌었는지 리포트가 본다."""

    name: str
    sha256: str
    """items.jsonl 의 sha256 (16진수)."""
    counts: Mapping[str, int]
    """고른 종류 → 문항 수."""


class StoredSearch:
    """문항마다 세트에 저장된 상위 5 청크를 검색 결과로 돌려준다. 답변 경로를 부르기 전에 문항마다 use() 한다.

    예)
        search = StoredSearch()
        run = AnswerRun(search, make_answerer, before_answer=lambda item: search.use(items[item.id]))
    """

    def __init__(self) -> None:
        self._item: F4Item | None = None

    def use(self, item: F4Item) -> None:
        """다음 검색에 돌려줄 문항을 고른다."""
        self._item = item

    async def search(self, query: str) -> SearchResult:
        """use() 로 고른 문항의 저장 문맥.

        Raises:
          RuntimeError: use() 를 부르지 않았을 때.
          ValueError: 질문이 고른 문항의 질문과 다를 때 (다른 문항의 문맥을 넘기지 않게).
        """
        if self._item is None:
            raise RuntimeError("call use() with the item to answer before search")
        if query != self._item.question:
            raise ValueError(f"query does not match the item in use: {self._item.id=}")
        return self._item.search_result()


def load_f4_items(
    bench_data: Path, kinds_by_set: Mapping[str, frozenset[str]]
) -> tuple[list[F4Item], list[F4Set]]:
    """세트마다 고른 종류의 문항 (세트 순서 → 파일 순서)과 읽은 세트 정보.

    Args:
      bench_data: LLM 레포의 bench/data 폴더 (세트 폴더가 있는 곳).
      kinds_by_set: 세트 폴더 이름 → 그 세트에서 고를 kind.

    Raises:
      ValueError: 고른 종류가 F4_KINDS 밖이거나, 세트에 그 종류가 없거나, 문항 id 가 세트 사이에 겹칠 때.
    """
    items: list[F4Item] = []
    sets: list[F4Set] = []
    for name, kinds in kinds_by_set.items():
        if unknown := kinds - set(F4_KINDS):
            raise ValueError(f"{name}: kinds not in F4_KINDS: {sorted(unknown)}")
        path = bench_data / name / "items.jsonl"
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
        picked = [F4Item.from_json(row, name) for row in rows if row["kind"] in kinds]
        counts = {kind: sum(item.kind == kind for item in picked) for kind in sorted(kinds)}
        if missing := [kind for kind, count in counts.items() if not count]:
            raise ValueError(f"{name}: no items of kind {missing}")
        items += picked
        sets.append(F4Set(name=name, sha256=hashlib.sha256(raw).hexdigest(), counts=counts))
    if len({item.id for item in items}) != len(items):
        raise ValueError("F4 item ids must be unique across sets")
    return items, sets


def _hit(chunk: ContextChunk) -> Hit:
    # 청크 id 는 "{문서 id}-{순번}" (적재기 규칙). token_count 는 답변 경로가 쓰지 않아 0
    doc_id, seq = chunk.chunk_id.rsplit("-", 1)
    return Hit(
        chunk=Chunk(
            id=chunk.chunk_id,
            doc_id=doc_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            seq=int(seq),
            token_count=0,
        ),
        score=chunk.score,
    )
