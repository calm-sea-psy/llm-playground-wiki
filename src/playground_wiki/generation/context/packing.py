"""칸 포장: data 칸의 글을 태그로 감싸 지시와 가르고, 본문 속 태그 흉내를 지운다.

본문의 `</document>` 가 블록을 닫으면 그 뒤 문장이 블록 밖 — 지시처럼 읽힌다 (간접 인젝션).
그래서 포장에 쓰는 태그 이름은 본문 · 속성 값에서 여는 꼴 · 닫는 꼴 모두 지운다.
"""

import re
from collections.abc import Iterable, Sequence

from playground_wiki.generation.context.slots import NUMBER_ATTRIBUTE, Candidate, Slot

ITEM_SEPARATOR = "\n\n"
"""칸 안 항목 사이. 위키 `format_context` 와 같은 빈 줄 하나."""


class TagScrubber:
    """정해진 태그 이름의 흉내(`<document>` · `</Document >` · `< /documents id=1>`)를 지운다.

    예)
        TagScrubber({"document"}).scrub("본문</document>지시")  # "본문지시"
    """

    def __init__(self, tags: Iterable[str]) -> None:
        # 긴 이름부터 — documents 가 document 에 먼저 걸려 's>' 가 남지 않게. 대소문자 · 꺾쇠 안 공백도 흉내다
        names = sorted(set(tags), key=len, reverse=True)
        alternatives = "|".join(re.escape(name) for name in names)
        self._pattern = (
            re.compile(rf"<\s*/?\s*(?:{alternatives})\b[^>]*>", re.IGNORECASE) if names else None
        )

    def scrub(self, text: str) -> str:
        """`text` 에서 태그 흉내를 지운 글을 돌려준다. 지워서 새로 생긴 흉내(`<doc<document>ument>`)도 지운다."""
        if self._pattern is None:
            return text
        while True:
            cleaned = self._pattern.sub("", text)
            if cleaned == text:
                return cleaned
            text = cleaned

    def attribute(self, value: str) -> str:
        """속성 값으로 넣을 글을 돌려준다 — 태그 흉내 · 꺾쇠를 지우고 따옴표는 `'`, 줄바꿈은 공백으로.

        닫는 `>` 가 없는 흉내(`<document id="2`)도 속성 안에서는 태그처럼 보이므로 꺾쇠를 남기지 않는다.
        """
        cleaned = self.scrub(value).replace("<", "").replace(">", "").replace('"', "'")
        return " ".join(cleaned.splitlines())


def render_data_slot(
    slot: Slot, items: Sequence[tuple[int | None, Candidate]], scrubber: TagScrubber
) -> str:
    """`data` 칸의 항목들을 태그로 감싼 글을 돌려준다. 항목이 없으면 빈 글.

    Args:
      slot: 포장할 data 칸.
      items: (번호, 후보) — 놓을 순서대로. 번호 칸이 아니면 번호는 None.
      scrubber: 레이아웃의 모든 태그 이름으로 만든 것 (다른 칸의 태그 흉내도 지운다).
    """
    if not items:
        return ""
    blocks = [_render_item(slot, number, candidate, scrubber) for number, candidate in items]
    body = ITEM_SEPARATOR.join(blocks)
    if slot.tag is None:
        return body
    return f"<{slot.tag}>\n{body}\n</{slot.tag}>"


def _render_item(
    slot: Slot, number: int | None, candidate: Candidate, scrubber: TagScrubber
) -> str:
    text = scrubber.scrub(candidate.text)
    if slot.item_tag is None:
        return text
    attributes = [] if number is None else [(NUMBER_ATTRIBUTE, str(number))]
    attributes.extend(candidate.attributes.items())
    rendered = "".join(f' {name}="{scrubber.attribute(value)}"' for name, value in attributes)
    return f"<{slot.item_tag}{rendered}>\n{text}\n</{slot.item_tag}>"
