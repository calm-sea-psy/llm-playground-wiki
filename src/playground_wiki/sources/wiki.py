"""위키백과 XML 덤프를 스트리밍으로 읽어 정제된 문서를 내놓는 코퍼스."""

import bz2
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import quote

from lxml import etree
from playground_corpus import Document

from playground_wiki.ingest.cleaner import WikiCleaner

logger = logging.getLogger(__name__)

SOURCE_NAME = "kowiki"
URL_PREFIX = "https://ko.wikipedia.org/wiki/"
ARTICLE_NAMESPACE = "0"
MIN_CHARS = 200  # 정제 후 본문이 이보다 짧은 stub 은 검색 근거로 쓸 내용이 거의 없다
REDIRECT_MARKERS = ("#넘겨주기", "#redirect")
POOL_CHUNKSIZE = 32  # 문서 하나 정제가 수 ms 라 프로세스 간 전달 비용을 묶어서 줄인다


@dataclass(frozen=True)
class RawPage:
    """덤프에서 읽은 정제 전 문서."""

    id: str
    title: str
    wikitext: str


@dataclass(frozen=True)
class Redirect:
    """넘겨주기 문서. 옛 제목 → 현재 제목 매칭에 쓴다."""

    title: str
    target: str


@dataclass(frozen=True)
class Skipped:
    """정제 후 걸러진 문서와 이유."""

    title: str
    reason: str
    """`disambiguation` 또는 `stub`."""


type ParsedPage = Document | Redirect | Skipped


class WikiSource:
    """위키 덤프의 본문 문서(ns 0)를 정제해 내놓는다. 넘겨주기 · 동음이의 · stub 은 뺀다.

    예)
        source = WikiSource(Path("data/dump/kowiki-20261001-pages-articles.xml.bz2"), dump_date="20261001")
        for doc in source.documents():
            ...
    """

    def __init__(
        self,
        dump_path: Path,
        *,
        dump_date: str,
        cleaner: WikiCleaner | None = None,
        min_chars: int = MIN_CHARS,
        workers: int | None = None,
    ) -> None:
        self._dump_path = dump_path
        self._dump_date = dump_date
        self._cleaner = cleaner or WikiCleaner()
        self._min_chars = min_chars
        self._workers = workers or max(1, (os.cpu_count() or 2) - 2)

    def documents(self) -> Iterator[Document]:
        """정제된 본문 문서를 덤프 순서대로 돌려준다."""
        for page in self.parsed():
            if isinstance(page, Document):
                yield page

    def parsed(self) -> Iterator[ParsedPage]:
        """본문 문서 · 넘겨주기 · 걸러진 문서를 덤프 순서대로 돌려준다.

        정제는 여러 프로세스에서 하고, 결과 순서는 덤프 순서를 지킨다.
        """
        process = partial(
            parse_page,
            cleaner=self._cleaner,
            dump_date=self._dump_date,
            min_chars=self._min_chars,
        )
        with Pool(self._workers) as pool:
            yield from pool.imap(process, self.raw_pages(), chunksize=POOL_CHUNKSIZE)

    def raw_pages(self) -> Iterator[RawPage]:
        """덤프에서 본문 이름공간(ns 0) 문서를 정제 전 상태로 돌려준다."""
        with bz2.open(self._dump_path, "rb") as f:
            namespace = ""
            for _, elem in etree.iterparse(f, events=("end",)):
                tag = elem.tag
                if not namespace and tag.endswith("}siteinfo"):
                    namespace = tag[: tag.index("}") + 1]
                if tag != f"{namespace}page":
                    continue
                page = _read_page(elem, namespace)
                _release(elem)
                if page is not None:
                    yield page


def _read_page(elem: etree._Element, namespace: str) -> RawPage | None:
    if elem.findtext(f"{namespace}ns") != ARTICLE_NAMESPACE:
        return None
    title = elem.findtext(f"{namespace}title") or ""
    page_id = elem.findtext(f"{namespace}id") or ""
    revision = elem.find(f"{namespace}revision")
    text = revision.findtext(f"{namespace}text") if revision is not None else None
    redirect = elem.find(f"{namespace}redirect")
    if redirect is not None:
        # 넘겨주기는 본문 대신 대상 제목을 wikitext 자리에 표시해 넘긴다
        return RawPage(
            id=page_id, title=title, wikitext=f"#넘겨주기 [[{redirect.get('title', '')}]]"
        )
    return RawPage(id=page_id, title=title, wikitext=text or "")


def _release(elem: etree._Element) -> None:
    """처리한 page 요소와 앞서 처리한 형제 요소를 메모리에서 지운다."""
    elem.clear()
    parent = elem.getparent()
    if parent is None:
        return
    while elem.getprevious() is not None:
        del parent[0]


def redirect_target(wikitext: str) -> str | None:
    """넘겨주기 문서면 대상 제목(절 이름은 뗀다), 아니면 None."""
    if not wikitext.lstrip()[:20].lower().startswith(REDIRECT_MARKERS):
        return None
    return wikitext.split("[[", 1)[-1].split("]]", 1)[0].split("#", 1)[0].strip()


def parse_page(
    page: RawPage, *, cleaner: WikiCleaner, dump_date: str, min_chars: int
) -> ParsedPage:
    """문서 하나를 넘겨주기 · 걸러진 문서 · 본문 문서 중 하나로 바꾼다. 정제 규칙 점검 스크립트도 이 판정을 쓴다."""
    if (target := redirect_target(page.wikitext)) is not None:
        return Redirect(title=page.title, target=target)
    cleaned = cleaner.clean(page.wikitext)
    if cleaned.is_disambiguation:
        return Skipped(title=page.title, reason="disambiguation")
    text = cleaned.text
    if len(text) < min_chars:
        return Skipped(title=page.title, reason="stub")
    return Document(
        id=page.id,
        title=page.title,
        text=text,
        sections=cleaned.sections,
        categories=cleaned.categories,
        meta={
            "source": SOURCE_NAME,
            "url": URL_PREFIX + quote(page.title.replace(" ", "_")),
            "dump_date": dump_date,
        },
    )
