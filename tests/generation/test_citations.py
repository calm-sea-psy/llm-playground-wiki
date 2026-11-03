from urllib.parse import quote

from playground_corpus import Chunk
from playground_retrieval.core import Hit

from playground_wiki.generation import resolve_citations, sources, wiki_url
from playground_wiki.generation.context_layouts import ask_documents
from playground_wiki.sources.wiki import URL_PREFIX


def _hit(title: str, section: str = "", seq: int = 0, score: float = 0.9) -> Hit:
    chunk = Chunk(
        id=f"1-{seq}", doc_id="1", title=title, section=section, text="본문", seq=seq, token_count=3
    )
    return Hit(chunk=chunk, score=score)


def test_wiki_url_matches_the_url_the_source_stores():
    # sources 와 generation 은 서로 import 하지 못해 주소 앞부분을 따로 적는다 — 같은 값이어야 한다
    assert wiki_url("서울 지하철") == URL_PREFIX + quote("서울_지하철")


def test_sources_are_numbered_like_the_context_blocks():
    hits = [_hit("경복궁", "역사", 0, 0.95), _hit("서울", "", 1, 0.6)]

    numbered = sources(hits)

    assert [(s.number, s.title, s.section, s.score) for s in numbered] == [
        (1, "경복궁", "역사", 0.95),
        (2, "서울", "", 0.6),
    ]
    context = ask_documents(hits)
    assert all(f'<document id="{s.number}" title="{s.title}">' in context for s in numbered)


def test_wiki_url_uses_underscores_and_percent_encoding():
    assert wiki_url("서울 지하철 2호선") == (
        "https://ko.wikipedia.org/wiki/%EC%84%9C%EC%9A%B8_%EC%A7%80%ED%95%98%EC%B2%A0_2%ED%98%B8%EC%84%A0"
    )


def test_valid_numbers_are_kept_and_listed_once_in_order():
    answer = resolve_citations("흥선대원군이 중건했다 [2]. 한양은 수도였다 [1][2].", count=3)

    assert answer.text == "흥선대원군이 중건했다 [2]. 한양은 수도였다 [1][2]."
    assert answer.cited == (2, 1)
    assert answer.dropped == ()


def test_out_of_range_numbers_are_dropped_with_their_space():
    answer = resolve_citations("문장 하나 [7]. 문장 둘 [0][1].", count=2)

    assert answer.text == "문장 하나. 문장 둘 [1]."
    assert answer.cited == (1,)
    assert answer.dropped == (7, 0)


def test_comma_lists_are_normalized_and_filtered():
    answer = resolve_citations("두 문서에서 [1, 3,9].", count=3)

    assert answer.text == "두 문서에서 [1][3]."
    assert answer.dropped == (9,)


def test_title_brackets_are_left_alone():
    # 번호가 아닌 대괄호는 모델이 규칙을 어긴 것이지만 여기서 고치지 않는다 (번호만 다룬다)
    answer = resolve_citations("출처 [경복궁] 와 [1].", count=1)

    assert answer.text == "출처 [경복궁] 와 [1]."
    assert answer.cited == (1,)


def test_no_context_drops_every_number():
    answer = resolve_citations("문서 없이 [1] 답함.", count=0)

    assert answer.text == "문서 없이 답함."
    assert answer.dropped == (1,)
