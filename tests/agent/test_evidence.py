from playground_corpus import Chunk
from playground_retrieval.core import Hit

from playground_wiki.agent.evidence import merge_hits, ranked_candidates, select_answer_hits
from playground_wiki.generation.context import WorkingMemory
from playground_wiki.generation.context_layouts import (
    AGENT_DOCUMENTS_BUDGET,
    DOCUMENTS,
    hit_evidence,
)

BUDGET = 8192


def _hit(chunk_id: str, score: float, text: str = "본문") -> Hit:
    chunk = Chunk(
        id=chunk_id,
        doc_id=chunk_id.split("-", maxsplit=1)[0],
        title=f"문서{chunk_id}",
        section="",
        text=text,
        seq=0,
        token_count=5,
    )
    return Hit(chunk=chunk, score=score)


def _memory(found: dict[str, list[Hit]]) -> WorkingMemory:
    memory = WorkingMemory("질문")
    for query, hits in found.items():
        memory = memory.add_evidence(hit_evidence(hits, query=query))
    return memory


def test_merge_hits_keeps_one_chunk_with_the_higher_score_in_its_first_place():
    merged = merge_hits([_hit("1-0", 0.5), _hit("2-0", 0.9)], [_hit("1-0", 0.8), _hit("3-0", 0.7)])

    assert [(hit.chunk.id, hit.score) for hit in merged] == [
        ("1-0", 0.8),
        ("2-0", 0.9),
        ("3-0", 0.7),
    ]


def test_ranked_candidates_drop_low_scores_and_put_every_query_top_first():
    memory = _memory(
        {
            "첫째": [_hit("1-0", 0.99), _hit("1-1", 0.98), _hit("1-2", 0.2)],
            "둘째": [_hit("2-0", 0.6)],
        }
    )

    candidates = ranked_candidates(memory, slot=DOCUMENTS, low=0.5)

    # low 미만은 빠지고 찾은 순서 그대로
    assert [c.key for c in candidates] == ["1-0", "1-1", "2-0"]
    by_key = {c.key: c.score for c in candidates}
    assert by_key["2-0"] > by_key["1-1"]  # 둘째 검색어의 1위가 첫째의 2위보다 먼저 뽑힌다


def test_select_answer_hits_keeps_each_query_top_when_the_budget_is_tight():
    long = "가" * (AGENT_DOCUMENTS_BUDGET * 3 // 2 // 3)  # 셋이면 칸 예산을 넘는 길이
    first = [_hit("1-0", 0.99, long), _hit("1-1", 0.98, long)]
    second = [_hit("2-0", 0.6, long)]
    memory = _memory({"첫째": first, "둘째": second})

    chosen = select_answer_hits(memory, merge_hits(first, second), low=0.5, budget=BUDGET)

    assert [hit.chunk.id for hit in chosen] == ["1-0", "2-0"]


def test_select_answer_hits_is_empty_when_everything_is_below_low():
    hits = [_hit("1-0", 0.1)]

    assert select_answer_hits(_memory({"첫째": hits}), hits, low=0.5, budget=BUDGET) == ()
