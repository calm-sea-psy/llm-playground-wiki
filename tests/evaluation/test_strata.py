from playground_corpus import Chunk
from playground_evalkit import KorQuADItem
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation import split_of
from playground_wiki.evaluation.answer_run import EvalItem
from playground_wiki.evaluation.strata import PoolRow, pool_items, pool_row, strata_rows


def _paragraphs_in(split: str, count: int) -> list[str]:
    keys = (f"문서{i}#0" for i in range(1000))
    return [key for key in keys if split_of(key) == split][:count]


def _item(qid: str, title: str, answer: str) -> KorQuADItem:
    return KorQuADItem(id=qid, title=title, question=f"{qid}?", answer=answer, source_title=title)


def test_pool_items_keep_every_question_of_the_split_without_corpus_filters():
    eval_a, eval_b = _paragraphs_in("eval", 2)
    (bench,) = _paragraphs_in("bench", 1)
    items = [_item("c", "다", "정답"), _item("a", "가", "정답"), _item("b", "가", "옛 정답")]
    items.append(_item("d", "나", "정답"))
    paragraphs = {"a": eval_a, "b": eval_a, "c": eval_b, "d": bench}

    picked = pool_items(items, paragraphs, "eval")

    # 같은 문단의 질문 둘(a · b)도, 문서가 없는 질문(c)도 남는다
    assert [item.id for item in picked] == ["a", "b", "c"]


def test_pool_row_records_corpus_match_and_top_score():
    item = _item("a", "가", "정답")
    chunk = Chunk(
        id="1-0", doc_id="1", title="가", section="", text="가 > \n정답이다", seq=0, token_count=3
    )
    result = SearchResult(hits=(Hit(chunk=chunk, score=0.9),), took_ms=12.0)

    row = pool_row(item, "가#0", "본문에 새 값만 있다", result)
    empty = pool_row(item, "가#0", None, SearchResult(hits=(), took_ms=1.0))

    assert (row.in_corpus, row.answer_in_document) == (True, False)
    assert (row.top_score, row.answer_rank) == (0.9, 1)
    assert EvalItem.from_json(row.to_json()).gold_answer == "정답"  # 표본 줄로도 읽힌다
    assert (empty.in_corpus, empty.top_score, empty.answer_rank) == (False, None, None)


def test_pool_row_json_keeps_the_key_order_and_reads_back():
    row = _pool_row("a", "가#0", None)

    assert list(row.to_json()) == [
        "id",
        "source",
        "question",
        "gold_title",
        "gold_answer",
        "paragraph",
        "in_corpus",
        "answer_in_document",
        "top_score",
        "answer_rank",
        "took_ms",
    ]
    assert PoolRow.from_json(row.to_json()) == row


def _pool_row(qid: str, paragraph: str, score: float | None) -> PoolRow:
    return PoolRow(
        id=qid,
        source="korquad",
        question=f"{qid}?",
        gold_title="가",
        gold_answer="정답",
        paragraph=paragraph,
        in_corpus=True,
        answer_in_document=True,
        top_score=score,
        answer_rank=1,
        took_ms=1.0,
    )


def _strata_pool() -> list[PoolRow]:
    eval_a, eval_b = _paragraphs_in("eval", 2)
    bench_a, bench_b = _paragraphs_in("bench", 2)
    (calibrate,) = _paragraphs_in("calibrate", 1)
    return [
        _pool_row("e1", eval_a, 0.85),
        _pool_row("e2", eval_b, 0.90),
        _pool_row("e3", eval_a, 0.99),
        _pool_row("e4", eval_b, 0.98),
        _pool_row("e5", eval_b, 0.97),
        _pool_row("e6", eval_a, 0.50),  # 0.83 미만 — 고르지 않는다
        _pool_row("e7", eval_a, None),  # 관문 · 검색 결과 없음
        _pool_row("b1", bench_a, 0.88),
        _pool_row("b2", bench_b, 0.86),
        _pool_row("b3", bench_a, 0.99),
        _pool_row("b4", bench_b, 0.995),
        _pool_row("b5", bench_a, 0.96),
        _pool_row("c1", calibrate, 0.87),  # calibrate 몫은 쓰지 않는다
        _pool_row("c2", calibrate, 0.99),
    ]


def test_strata_rows_match_the_high_band_to_the_middle_band_per_split():
    picked = strata_rows(_strata_pool(), seed=0)

    ids = [row.id for row in picked]
    assert ids == sorted(ids)
    assert {"e1", "e2", "b1", "b2"} <= set(ids)
    assert len(set(ids) & {"e3", "e4", "e5"}) == 2  # eval 중간 2 → eval 높음 2
    assert len(set(ids) & {"b3", "b4", "b5"}) == 2  # bench 중간 2 → bench 높음 2
    assert not set(ids) & {"e6", "e7", "c1", "c2"}
    assert picked == strata_rows(_strata_pool(), seed=0)


def test_strata_rows_fill_the_middle_band_from_eval_first_and_keep_earlier_picks():
    rows = _strata_pool()

    two = {row.id for row in strata_rows(rows, seed=0, middle_size=2)}
    three = {row.id for row in strata_rows(rows, seed=0, middle_size=3)}

    assert len(two) == 4
    assert {"e1", "e2"} <= two  # eval 중간을 먼저 다 담는다
    assert not two & {"b1", "b2", "b3", "b4", "b5"}
    assert len(three) == 6
    assert two < three  # 늘려도 앞서 고른 문항이 남는다
    assert len(three & {"b1", "b2"}) == 1
    assert len(three & {"b3", "b4", "b5"}) == 1
