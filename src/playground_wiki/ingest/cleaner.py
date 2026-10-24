"""위키 마크업을 섹션별 평문으로 바꾸는 정제기."""

import re
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass

import mwparserfromhell
from mwparserfromhell.nodes import Heading, Node, Tag, Template, Wikilink
from mwparserfromhell.wikicode import Wikicode
from playground_corpus import Section

CATEGORY_PREFIXES = ("분류:", "category:")
MEDIA_PREFIXES = ("파일:", "file:", "그림:", "image:", "미디어:", "media:")

# 본문 검색에 도움이 안 되는 꼬리 섹션. 하위 섹션까지 함께 뺀다
DROP_SECTIONS = frozenset(
    {
        "각주",
        "주석",
        "내용주",
        "출처",
        "참고 문헌",
        "참고문헌",
        "참고 자료",
        "참고자료",
        "외부 링크",
        "외부링크",
        "같이 보기",
        "함께 보기",
        "관련 항목",
        "관련항목",
    }
)

# 내용째 버리는 태그. table 은 셀 단위 문장이 깨져 검색 품질을 깎으므로 통째로 뺀다.
# includeonly 는 다른 문서에 끼워 넣을 때만 나오고 그 문서 화면에는 안 나온다
INCLUDEONLY_TAG = "includeonly"
REF_TAG = "ref"
DROP_TAGS = frozenset(
    {
        REF_TAG,
        "references",
        "gallery",
        "chem",
        "ce",
        "timeline",
        "score",
        "syntaxhighlight",
        "source",
        "imagemap",
        "templatestyles",
        "mapframe",
        "maplink",
        "graph",
        "hiero",
        "table",
        INCLUDEONLY_TAG,
    }
)

# 위키 전처리기가 안을 읽지 않고 넘기는 확장 태그. 이 안의 "<!--" · "<ref>" 는 주석 · 각주가 아니라 글자다
EXTENSION_TAGS = (DROP_TAGS - {"table"}) | {
    "nowiki",
    "pre",
    "math",
    "poem",
    "categorytree",
    "inputbox",
    "section",
    "templatedata",
    "indicator",
}
LITERAL_TAGS = frozenset({"nowiki", "pre"})  # 안의 마크업을 글자 그대로 보여 주는 태그

# 닫히지 않은 숨김 요소를 위키가 숨기는 범위 (ko.wikipedia parse API 로 확인). span 은 문단 끝까지,
# div · p · blockquote · small 같은 나머지는 문서 끝까지. 문단 범위로 잘못 넣으면 숨은 글이 남으므로 확인한 것만 둔다
PARAGRAPH_HIDDEN_TAGS = frozenset({"span"})

# 수식은 문장 사이의 짧은 인라인 수식만 LaTeX 원문으로 남긴다. 지우면 "어떤 모수 로 결정되는" 처럼 문장이 깨지고,
# 긴 블록 수식은 임베딩 · BM25 에 잡음만 더한다. 40자면 L(\theta) · \frac{a}{b} · \sum_{i=1}^n x_i 정도까지 들어간다
MATH_TAG = "math"
MAX_INLINE_MATH_CHARS = 40

# 둘째 위치 인자가 원어 표기인 틀. 셋째는 한글 전사라 뺀다 ({{llang|en|Leslie Cheung|레슬리 청}} → Leslie Cheung)
# 둘째가 비면(안쪽 문자 전용 틀 {{MongolUnicode|…}} 가 지워진 경우) 다음 비지 않은 인자 (로마자 표기 등)
ORIGINAL_TEXT_TEMPLATES = frozenset({"lang", "llang", "langnf"})
ORIGINAL_TEXT_INDEX = 1  # 0 은 언어 코드

# 마지막 위치 인자가 본문에 드러나는 글자인 틀 ({{lang-en|Seoul}} · {{국기|가나}} · {{transl|grc|Isaak}})
KEEP_LAST_PARAM_TEMPLATES = frozenset(
    {"국기", "국기나라", "깃발나라", "nowrap", "transl", "transliteration"}
)

# 첫 위치 인자가 본문 글자인 틀 ({{한자|大韓民國}} · {{학명|Hordeum vulgare}} · {{ruby-ja|同人誌|どうじんし}})
FIRST_PARAM_TEMPLATES = frozenset(
    {
        "한자",
        "한국 한자",
        "학명",
        "ruby",
        "ruby-ja",
        "후리가나",
        "후리가나2",
        "중세 국어",
        "히브리어 이름",
        "임시링크",  # {{임시링크|표시 제목|언어|원어 제목}}
    }
)

# 위치 인자를 이어 붙여 한 단어로 보이는 틀 ({{linktext|行星|分化}} → 行星分化)
JOIN_PARAMS_TEMPLATES = frozenset({"linktext"})

# 이름표를 붙여 첫 위치 인자를 보이는 틀 ({{본명|전해림}} → 본명 전해림)
LABELED_TEMPLATES = frozenset({"본명", "문화어", "표준어"})

# 중국어 표기 틀은 이름 인자로 받는다. 번체 → 원문 → 간체 순으로 있는 것 하나
CHINESE_TEMPLATE_PREFIXES = ("zh", "lang-zh")
CHINESE_TEXT_KEYS = ("t", "c", "s")

# 첫 세 위치 인자가 연 · 월 · 일인 날짜 틀. KorQuAD 에 생몰일 질문이 많아 날짜를 살린다
DATE_TEMPLATES = frozenset(
    {
        "출생일",
        "출생일과 나이",
        "사망일",
        "사망일과 나이",
        "birth date",
        "birth date and age",
        "death date",
        "death date and age",
    }
)

# 함선 이름 틀. {{USS|폼프렛|SS-391}} → USS 폼프렛 (접두 + 첫 위치 인자)
SHIP_PREFIX_TEMPLATES = frozenset({"uss", "hms", "hmas", "hmcs", "sms", "rms", "ins", "jds"})

DISAMBIGUATION_MARKERS = ("동음이의", "disambig")

_INTERLANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z]+)*:")
_LEFTOVER_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_LEFTOVER_LINK = re.compile(
    r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]"
)  # 파서가 놓친 [[대상|표시]] → 표시
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")  # 닫히지 않아 파서가 놓친 태그
_BROKEN_REF = re.compile(r"<ref\b[^>\n]*$", re.MULTILINE)  # 따옴표가 안 닫혀 줄 끝까지 이어진 각주
_TABLE_LINE = re.compile(r"^[ \t]*(?:\{\||\|\}|\|-|\||!).*$", re.MULTILINE)  # 파서가 놓친 표 줄
_LIST_MARKER = re.compile(
    r"^[ \t]*[*#:;]+[ \t]*", re.MULTILINE
)  # 앞의 수식을 지우면 공백 뒤에 남는다
_MAGIC_WORD = re.compile(r"__[A-Z]+__")  # __NOTOC__ · __TOC__ 등
_FOOTNOTE_MARK = re.compile(r"\[(?:\d+|주 \d+|각주 \d+)\]")
# 표 시작 · 끝({| · |})은 _TABLE_LINE 이 줄 단위로 지운다. 줄 중간의 {| · |} 는 LaTeX(\frac{1+|z|}{…}) 라 남긴다
_LEFTOVER_BRACKETS = re.compile(r"\[\[|\]\]|\{\{|\}\}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,])")  # 지운 각주 · 틀 자리에 남은 공백
_EMPTY_PARENS = re.compile(r"[(（]\s*[,;、]?\s*[)）]")
# 괄호 첫머리 · 끝의 구분자 — 발음 기호 틀({{IPA|…}})처럼 일부러 지운 틀 자리에 남는다: "(, X)" · "(; X; )"
_PAREN_LEADING_SEP = re.compile(r"([(（])[ \t]*[,;、][ \t]*")
_PAREN_TRAILING_SEP = re.compile(r"[ \t]*(?<!\\)[,;、][ \t]*([)）])")  # LaTeX \, (얇은 공백)는 둔다
_SPACES = re.compile(r"[ \t\u00a0\u200b]+")  # 공백 · 탭 · NBSP · zero-width space
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")

# 위키 전처리기가 앞에서부터 읽으며 짝을 찾는 자리: 주석 시작과 여는 태그 (끝의 / 는 스스로 닫힌 태그)
_PREPROCESSOR_MARK = re.compile(r"<!--|<([a-zA-Z][a-zA-Z0-9]*)(?:\s[^<>]*?)?(/?)>")
_REF_OPEN = re.compile(r"<ref\b", re.IGNORECASE)
# 위키 화면에 안 그려지는 style. hidden 속성은 위키가 지우고 글을 보여 주므로 넣지 않는다 (parse API 로 확인)
_STYLE_ATTR = re.compile(r"""(?<![\w-])style\s*=\s*("[^"]*"|'[^']*'|[^\s"'>]+)""", re.IGNORECASE)
_HIDDEN_STYLE = re.compile(
    r"(?<![\w-])(?:display\s*:\s*none\b|visibility\s*:\s*hidden\b|font-size\s*:\s*0(?:\.0*)?(?![\d.]))",
    re.IGNORECASE,
)
_OPEN_TAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)(\s[^<>]*)>")
# 숨는 범위 표지. 덤프 XML 에는 제어 문자가 못 들어가므로 원문과 겹치지 않는다
_HIDE_PARAGRAPH = "\x01"
_HIDE_REST = "\x02"
_PARAGRAPH_BREAK = re.compile(r"\n(?:[ \t]*\n|(?=[=*#:;]))")  # 빈 줄 · 제목 줄 · 목록 줄 앞


@dataclass(frozen=True)
class CleanedPage:
    """정제한 문서 한 건의 본문 구조."""

    sections: tuple[Section, ...]
    """본문이 있는 섹션만, 문서 순서대로. 첫머리는 level 0."""
    categories: tuple[str, ...]
    is_disambiguation: bool
    """동음이의 틀이나 분류가 붙은 문서."""

    @property
    def text(self) -> str:
        """섹션 본문을 빈 줄로 이은 전체 본문."""
        return "\n\n".join(section.text for section in self.sections)


class WikiCleaner:
    """위키 마크업에서 틀 · 각주 · 표 · 파일 링크 · 화면에 안 보이는 글을 걷어 내고 섹션 구조를 남긴다.

    화면에 안 보이는 글이 남으면 사람은 못 보는 지시문이 검색 청크로 들어가 간접 인젝션 통로가 된다.

    예)
        page = WikiCleaner().clean(wikitext)
        page.sections[1].title  # "역사"
    """

    def clean(self, wikitext: str) -> CleanedPage:
        """위키 마크업을 정제한 섹션 목록과 분류로 바꾼다."""
        code = mwparserfromhell.parse(_cut_unclosed(wikitext))
        is_disambiguation = _has_disambiguation_template(code)
        categories = _pop_links(code)
        _drop_tags(code)
        _rewrite_templates(code)
        if _mark_unclosed_hidden(code):  # 드물다. 표지를 넣은 문서만 잘라 다시 읽는다
            code = mwparserfromhell.parse(_cut_hidden(str(code)))
        is_disambiguation = is_disambiguation or any(
            marker in category for category in categories for marker in DISAMBIGUATION_MARKERS
        )
        return CleanedPage(
            sections=self._sections(code),
            categories=tuple(dict.fromkeys(categories)),
            is_disambiguation=is_disambiguation,
        )

    def _sections(self, code: Wikicode) -> tuple[Section, ...]:
        sections: list[Section] = []
        drop_below: int | None = None  # 빼는 섹션의 레벨. 그보다 깊은 하위 섹션도 뺀다
        for part in code.get_sections(flat=True, include_lead=True):
            nodes = part.nodes
            if nodes and isinstance(nodes[0], Heading):
                level = nodes[0].level
                title = clean_text(nodes[0].title.strip_code())
                body = Wikicode(nodes[1:])
            else:
                level, title, body = 0, "", part
            if drop_below is not None and level > drop_below:
                continue
            drop_below = level if title in DROP_SECTIONS else None
            if drop_below is not None:
                continue
            text = clean_text(body.strip_code(normalize=True, collapse=True))
            if text:
                sections.append(Section(level=level, title=title, text=text))
        return tuple(sections)


def clean_text(text: str) -> str:
    """strip_code 뒤에 남은 마크업 조각과 공백을 정리한다."""
    previous = None
    while previous != text:  # 안쪽부터 한 겹씩 남은 틀을 걷는다
        previous = text
        text = _LEFTOVER_TEMPLATE.sub("", text)
    text = _LEFTOVER_LINK.sub(r"\1", text)
    text = _BROKEN_REF.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _TABLE_LINE.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _MAGIC_WORD.sub("", text)
    text = _FOOTNOTE_MARK.sub("", text)
    text = _LEFTOVER_BRACKETS.sub("", text)
    text = _PAREN_LEADING_SEP.sub(r"\1", text)
    text = _PAREN_TRAILING_SEP.sub(r"\1", text)
    text = _EMPTY_PARENS.sub("", text)  # 구분자를 걷은 뒤 빈 괄호 "(; )" → "()" 를 지운다
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _cut_unclosed(wikitext: str) -> str:
    """위키 전처리기가 닫히지 않은 채로 숨기는 범위를 원문에서 지운다.

    위키처럼 앞에서부터 짝을 찾는다. 닫히지 않은 주석 · includeonly 는 문서 끝까지 숨는다.
    닫는 태그 없는 ref 는 뒤 ref 의 </ref> 까지 삼킨다 — 파서는 그 사이 글을 본문으로 남긴다.
    뒤에 </ref> 가 아예 없으면 위키도 글자로 보여 주므로 둔다.
    """
    text, pos = wikitext, 0
    while match := _PREPROCESSOR_MARK.search(text, pos):
        if match.group(0) == "<!--":
            end = text.find("-->", match.end())
            if end == -1:
                return text[: match.start()]
            pos = end + len("-->")
            continue
        name = match.group(1).lower()
        self_closing = match.group(2) == "/"  # <ref name="a" /> 처럼 짝이 필요 없다
        if name not in EXTENSION_TAGS or self_closing:
            pos = match.end()
            continue
        close = re.compile(rf"</{name}\s*>", re.IGNORECASE).search(text, match.end())
        if close is None:
            if name == INCLUDEONLY_TAG:
                return text[: match.start()]
            pos = match.end()
            continue
        if name == REF_TAG and _REF_OPEN.search(text, match.end(), close.start()):
            text = text[: match.start()] + text[close.end() :]
            pos = match.start()
            continue
        pos = close.end()
    return text


def _mark_unclosed_hidden(code: Wikicode) -> bool:
    """파서가 짝을 못 찾아 글자로 남긴 숨김 요소의 여는 태그를 숨는 범위 표지로 바꾼다. 바꾼 게 있으면 True."""
    marked = False
    for node in code.filter_text(recursive=True):
        if "<" not in node.value or not any(
            _hides(match.group(2)) for match in _OPEN_TAG.finditer(node.value)
        ):
            continue
        if any(_tag_name(ancestor) in LITERAL_TAGS for ancestor in code.get_ancestors(node)):
            continue  # nowiki · pre 안의 태그는 화면에 글자로 보인다
        node.value = _OPEN_TAG.sub(_hide_mark, node.value)
        marked = True
    return marked


def _hide_mark(match: re.Match[str]) -> str:
    if not _hides(match.group(2)):
        return match.group(0)
    return _HIDE_PARAGRAPH if match.group(1).lower() in PARAGRAPH_HIDDEN_TAGS else _HIDE_REST


def _cut_hidden(wikitext: str) -> str:
    """숨는 범위 표지부터 위키가 숨기는 데까지 지운다. 문단 표지는 문단 끝까지, 나머지는 문서 끝까지."""
    rest = wikitext.find(_HIDE_REST)
    if rest != -1:
        wikitext = wikitext[:rest]
    while (start := wikitext.find(_HIDE_PARAGRAPH)) != -1:
        end = _PARAGRAPH_BREAK.search(wikitext, start)
        wikitext = wikitext[:start] + (wikitext[end.start() :] if end else "")
    return wikitext


def _hides(attributes: str) -> bool:
    """태그 속성 문자열의 style 이 요소를 화면에서 숨기는지."""
    style = _STYLE_ATTR.search(attributes)
    return style is not None and _HIDDEN_STYLE.search(style.group(1)) is not None


def _tag_name(node: Node) -> str:
    return str(node.tag).strip().lower() if isinstance(node, Tag) else ""


def _has_disambiguation_template(code: Wikicode) -> bool:
    for template in code.ifilter_templates(recursive=True):
        name = str(template.name).strip().lower()
        if any(marker in name for marker in DISAMBIGUATION_MARKERS):
            return True
    return False


def _pop_links(code: Wikicode) -> list[str]:
    """분류 링크를 빼서 이름을 돌려주고, 파일 · 언어 간 링크는 지운다."""
    categories: list[str] = []
    for link in code.filter_wikilinks(recursive=True):
        target = str(link.title).strip()
        lowered = target.lower()
        if lowered.startswith(CATEGORY_PREFIXES):
            categories.append(target.split(":", 1)[1].strip())
            _remove(code, link)
        elif lowered.startswith(MEDIA_PREFIXES) or _is_interlanguage(target):
            _remove(code, link)
    return categories


def _drop_tags(code: Wikicode) -> None:
    for tag in code.filter_tags(recursive=True):
        name = str(tag.tag).strip().lower()
        if name in DROP_TAGS or _hides("".join(str(attribute) for attribute in tag.attributes)):
            _remove(code, tag)
        elif name == MATH_TAG:
            _replace(code, tag, _inline_math(tag))


def _inline_math(tag: Tag) -> str:
    """짧은 인라인 수식은 LaTeX 원문 한 줄로, 블록 수식 · 긴 수식은 빈 문자열로."""
    if tag.has("display") and str(tag.get("display").value).strip().lower() == "block":
        return ""
    latex = " ".join(str(tag.contents).split())
    return latex if len(latex) <= MAX_INLINE_MATH_CHARS else ""


def _rewrite_templates(code: Wikicode) -> None:
    """글자가 드러나는 틀은 그 글자로 바꾸고 나머지 틀은 지운다."""
    # 안쪽 틀부터 바꾼다. 바깥부터 바꾸면 {{llang|ja|{{ruby-ja|西東京市|…}}}} 의 안쪽 틀이 원문째 남았다가 지워진다
    for template in reversed(code.filter_templates(recursive=True)):
        name = str(template.name).strip().lower()
        replacement = _template_text(name, template)
        _replace(code, template, replacement)


def _template_text(name: str, template: Template) -> str:
    positional = [str(param.value).strip() for param in template.params if not param.showkey]
    named = {
        str(param.name).strip(): str(param.value).strip()
        for param in template.params
        if param.showkey
    }
    if name in DATE_TEMPLATES:
        return _format_date(positional[:3])
    if name.startswith(CHINESE_TEMPLATE_PREFIXES):
        return next((named[key] for key in CHINESE_TEXT_KEYS if named.get(key)), "")
    if name == "vie":
        return named.get("v", "")
    if not positional:
        return ""
    if name in ORIGINAL_TEXT_TEMPLATES:
        if len(positional) <= ORIGINAL_TEXT_INDEX:
            return positional[-1]
        return next((p for p in positional[ORIGINAL_TEXT_INDEX:] if p), "")
    if name in JOIN_PARAMS_TEMPLATES:
        return "".join(positional)
    if name in KEEP_LAST_PARAM_TEMPLATES or name.startswith("lang-"):
        return positional[-1]
    if name in FIRST_PARAM_TEMPLATES:
        return positional[0]
    if name in LABELED_TEMPLATES:
        return f"{name} {positional[0]}"
    if name in SHIP_PREFIX_TEMPLATES:
        return f"{name.upper()} {positional[0]}"
    return ""


def _format_date(parts: Sequence[str]) -> str:
    units = ("년", "월", "일")
    pieces = [
        f"{part.lstrip('0') or '0'}{unit}"
        for part, unit in zip(parts, units, strict=False)
        if part.isdigit()
    ]
    return " ".join(pieces)


def _is_interlanguage(target: str) -> bool:
    # [[en:Seoul]] · [[zh-yue:首爾]] 처럼 언어 코드 접두가 붙은 링크
    return _INTERLANGUAGE.match(target) is not None


def _remove(code: Wikicode, node: Wikilink | Tag) -> None:
    with suppress(ValueError):  # 이미 지운 노드 안에 있던 노드
        code.remove(node)


def _replace(code: Wikicode, node: Template | Tag, value: str) -> None:
    with suppress(ValueError):  # 이미 지운 노드 안에 있던 노드
        code.replace(node, value)
