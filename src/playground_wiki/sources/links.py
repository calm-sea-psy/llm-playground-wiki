"""위키텍스트의 문서 링크로 문서마다 역링크 수를 센다 — 덤프에 조회 수가 없어 인기도를 이것으로 대신한다."""

import re
from collections import Counter
from collections.abc import Iterable, Mapping

LINK = re.compile(r"\[\[([^\[\]|#\n]*)")
"""`[[대상]]` · `[[대상|보이는 글]]` · `[[대상#절]]` 의 대상 부분."""
# 넘겨주기를 다시 넘겨주기로 가리키는 사슬은 2~3단이면 끝난다. 고리에서 멈추려는 상한
MAX_REDIRECT_HOPS = 5


def normalize_title(title: str) -> str:
    """링크 대상과 문서 제목을 같은 꼴로: 밑줄은 공백, 공백은 하나로, 앞 ':' 를 떼고 첫 글자는 대문자 (미디어위키 규칙)."""
    title = " ".join(title.replace("_", " ").split()).lstrip(":").strip()
    return title[:1].upper() + title[1:]


def article_links(wikitext: str) -> set[str]:
    """문서가 가리키는 링크 대상 (정규화, 중복 없음). 분류 · 파일 · 다른 언어 링크도 섞여 있다 — 셀 때 본문 문서 제목으로 거른다."""
    return {target for match in LINK.findall(wikitext) if (target := normalize_title(match))}


class BacklinkCounter:
    """문서를 하나씩 받아 대상마다 그것을 가리킨 문서 수를 센다. 넘겨주기는 최종 대상으로 넘겨 센다.

    별칭(넘겨주기)이 많은 인기 문서가 링크를 별칭 이름으로 받아도 제 몫으로 세어진다.
    틀(내비게이션 상자 등)이 펼치는 링크는 원문 위키텍스트에 없어 세지 않는다.

    예)
        counter = BacklinkCounter({"서울시": "서울특별시"})
        counter.add("대한민국", "수도는 [[서울시|서울]]이다.")
        counter.counts(["서울특별시"])  # {"서울특별시": 1}
    """

    def __init__(self, redirects: Mapping[str, str]) -> None:
        self._redirects = {normalize_title(k): normalize_title(v) for k, v in redirects.items()}
        self._links: Counter[str] = Counter()

    def resolve(self, title: str) -> str:
        """넘겨주기를 따라간 최종 제목 (정규화한 제목을 받는다). 넘겨주기가 아니면 그대로."""
        for _ in range(MAX_REDIRECT_HOPS):
            target = self._redirects.get(title)
            if target is None or target == title:
                return title
            title = target
        return title

    def add(self, title: str, wikitext: str) -> None:
        """문서 하나의 링크를 센다. 넘겨주기를 넘긴 뒤 같은 대상은 한 번, 자기 자신은 뺀다."""
        targets = {self.resolve(target) for target in article_links(wikitext)}
        targets.discard(self.resolve(normalize_title(title)))
        self._links.update(targets)

    def counts(self, titles: Iterable[str]) -> dict[str, int]:
        """titles(본문 문서 제목)마다 역링크 수. 본문 문서가 아닌 대상(분류 · 없는 문서)은 여기서 버려진다."""
        return {title: self._links[normalize_title(title)] for title in titles}
