"""반사실 문항 규칙의 기준값과 정규식 — 값 치환과 사전 · 사후 검사가 함께 쓴다."""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, product
from types import MappingProxyType
from typing import Any, Self

SUBSTITUTE_ATTEMPTS = 40  # 숫자 바꾸기 재시도 — 바뀐 값이 원래 문맥에 없을 때까지
# 연도 치환은 원래 해에서 이 범위 안으로 — 넓으면 주변 사건과 시기가 어긋난다 (1992년 → 2003년 선고)
YEAR_WINDOW = 5
DECADE_WINDOW = 10  # 10년 단위 치환 범위 — 60년대 → 90년대 방지
CENTURY_SHIFT = 3  # 세기 치환 범위
CENTURY_MAX = 21  # 지금이 21세기 — 그 뒤 세기는 과거 사실로 말이 안 된다
NEAR_CHOICES = 12  # 가까운 후보 몇 개 중에서 고를지
NEIGHBOR_WINDOW = (
    200  # 답 근처로 볼 앞뒤 글자 수 — 같은 종류 값과의 순서 · 원래 값 잔존 · 연호를 여기서 본다
)
NAME_MAX_LEN = 10  # 사람 이름꼴 답으로 볼 최대 길이 (직함 섞인 긴 구절 배제)
NAME_PART_MIN = 2  # 바꾼 뒤 남았는지 볼 이름 어절의 최소 글자 수 (한 글자 어절은 흔해서 뺀다)
YEAR_MIN_DIGITS = 3  # 'N년' 이 연도인 최소 자리수 — 두 자리 이하(32년)는 기간
COUNT_MIN = 2  # 수를 바꿀 때 최솟값 — 0 · 1 은 '0개의 파열음 · 1번 반복'처럼 말이 안 된다
COUNT_MIN_CHANGE = 0.05  # 수를 바꿀 때 원래 값 대비 최소 차이 비율 — 1445유로 → 1444유로 방지
# 일을 바꿀 때 최소 차이 — 12월 13일 → 12월 14일 · 8월 19일 → 8월 16일 은 오타로 보인다
DAY_MIN_CHANGE = 5
YEAR_MIN_CHANGE = 2  # 연도를 바꿀 때 최소 차이 — 1998년 → 1999년 은 오타로 보인다
ROUND_KEEP_ZEROS = 2  # 딱 떨어지는 수의 끝 0 을 이만큼까지는 그대로 지킨다 (800만 → 860만 방지)
PERSON_CUE_WINDOW = 30  # 인물 이름 앞뒤로 성별 · 나이 단서를 찾을 글자 수
RELATIVE_WINDOW = 80  # 연도 답 앞뒤로 상대 시간 표현(이듬해 · 3년 뒤)을 찾을 글자 수
RANGE_GAP = 15  # 연도와 범위 잇는 말(~ · 부터) 사이에 올 수 있는 글자 수 ('2008년 2월 ~ 2009년')
# 같은 자리인지 가릴 때 볼 정답 앞뒤 글자 수 — 겹친 청크의 같은 문장을 한 자리로 센다
SPOT_WIDTH = 12
# 계산값을 볼 때 쓰는 다른 수의 상한 — 연도(1944)끼리 더한 값은 보지 않는다
COMPUTED_OPERAND_MAX = 1000
KOREAN_NAME_LEN = 3  # 가상 이름으로 바꿀 한국식 이름의 음절 수 — 성 + 두 글자
FOREIGN_NAME_WORDS = 2  # 가상 이름으로 바꿀 외국 이름의 어절 수 — 이름 + 성
# 인물을 이보다 많은 곳에서 바꾸면 뺀다 — 문서 주인공급이라 별칭 · 이니셜이 남는다 (전봉준 19곳 · 이명박 15곳).
# 검수를 마친 인물 문항은 3곳 이하였다
PERSON_REPLACE_MAX = 5
# 원래 수가 다른 단위로 남았는지 볼 최솟값 — 한 자리 수는 다른 사실(4마디 · 제2시민아파트)에 흔하다
NUMBER_LEFT_MIN = 10
# 예비 문항을 한 문단에서 쓸 수 — 하나면 예비 원본 후보가 50 을 겨우 채운다 (9차 초안 기준 18)
SPARE_PER_PARAGRAPH = 2
# 사람 검수에서 뺀 KorQuAD id — 원 문항 결함이거나, 문맥 밖 상식과 충돌해 규칙으로는 거를 수 없다
COUNTERFACTUAL_EXCLUDE = frozenset(
    {
        "6313886-0-1",  # 질문은 닉슨이 팻을 만난 날 · 문맥 날짜는 결혼일 (2026-10-28 검수)
        "6472235-1-1",  # 전대협 위원장 선고 — 서울형사지방법원은 1995년 통합, 옮길 수 있는 해가 없다 (10-29)
        "6530030-15-1",  # 바이오쇼크 E3 수상 — 2004년 개발 발표 전후로 옮길 해가 없다 (10-29)
        "6523631-2-1",  # 엑스박스 360 업데이트 2008 → 2004년 — 2005년 출시는 문맥 밖이라 순서로 못 본다 (10-29)
        "6125574-4-1",  # 블루스 코드 3가지 — 바로 뒤 문장이 I · IV · V 세 코드를 열거한다 (10-29)
    }
)
# 사람 검수(2026-10-28 ~ 29)에서 두 항목 모두 O 를 받은 값 — 다시 만들어도 이 값을 쓴다.
# 지금 규칙(연도 순서 · 월일 순서 · 이름 계열 · 이미 쓴 이름)을 어기면 쓰지 않고 새로 뽑는다
COUNTERFACTUAL_REVIEWED: Mapping[str, str] = MappingProxyType(
    {
        "5900104-2-0": "3개",
        "5933667-3-0": "1994년",  # 7차 — 1998년은 같은 청크의 '1997년 실내 시즌' 과 순서가 어긋났다
        "6097910-5-1": "3명",
        "6125574-14-0": "70년대 초반",
        "6125574-2-0": "6번",
        "6141330-2-0": "16-17세기",
        "6474741-2-1": "오전 12시 9분",
        "6488411-0-2": "29년",
        "6488411-14-0": "3갑",
        "6489313-11-2": "1523유로",
        "6490156-9-0": "2000년",
        "6506110-7-1": "2001년",
        "6527927-12-1": "8층",
        "6527927-4-2": "3번째",
        "6543321-1-0": "12월 3일",
        "6544134-1-2": "2010년",
        "6550225-2-0": "16만 7천명",
        "6581515-6-1": "6위",
        "6584295-13-1": "505명",
        "6584295-15-0": "3번",
        "6584295-28-0": "1942년",
        "6584295-42-0": "373대",
        "6584295-69-1": "1972년",
        # 인물 (6 · 7차 가상 이름)
        "6269760-43-1": "배동혁",
        "6545182-21-0": "염동훈",
        "6557695-3-2": "유경훈",
        "6575724-0-1": "후지타 마사키",
        "6577087-1-0": "권태석",
        "6584295-12-1": "스기모토 준이치",
        "6584295-31-1": "월터 리브스",
        "6584295-4-0": "기쿠치 아키라",
        "6599035-7-1": "리처드 홀트",
        # 예비 원본 (8차 검수)
        "6031331-3-2": "1992년",
        "6480874-1-1": "1952년",
        "6577228-1-0": "12월 10일",
        "6588681-0-0": "1938년",
        "6656752-0-0": "22승 9패",
        "6482346-0-1": "구민혁",
        "6543842-0-1": "엄재석",
        "6545182-10-2": "윤기태",
        "6581515-24-1": "황인규",
        # 예비 원본 (9차 검수 — 블루스 3가지 대신 고코곤 천황이 들어옴)
        "5912341-11-1": "6월 6일",
        "5912341-11-2": "6월 8일",
        "6031331-1-2": "90개",
        "6378640-1-1": "1968년",
        "6509164-0-2": "1388년",
        "6509164-2-1": "1356년 11월 9일",
        "6512125-12-0": "2억 3천만 유로",
        "6545197-2-2": "2018년",
        "6581515-24-0": "고승택",
    }
)
# 인물 반사실에 넣을 가상 이름 — 원래 이름의 표기 계열마다. 흔한 성 + 이름으로 짓고, 한국어 위키백과
# (2026-10-01 덤프) 문서 제목에 없는 이름만 골랐다 — 같은 이름의 실존 인물(임동규 · 최병철)과 겹치지 않게
FICTIONAL_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ko": (
            "권태석",
            "서진혁",
            "오창민",
            "유경훈",
            "조윤석",
            "황인규",
            "고승택",
            "구민혁",
            "배동혁",
            "성기원",
            "엄재석",
            "염동훈",
            "우성택",
            "원준호",
            "윤기태",
        ),
        "ja": (
            "오카모토 신지",
            "마쓰다 요시오",
            "후지타 마사키",
            "니시무라 고헤이",
            "기쿠치 아키라",
            "아오키 데쓰야",
            "스기모토 준이치",
            "우에노 겐지",
            "사카이 노부오",
            "오가와 히데오",
        ),
        "western": (
            "로버트 하딩",
            "에드워드 콜린스",
            "토머스 웰런",
            "리처드 홀트",
            "윌리엄 하퍼",
            "헨리 몰턴",
            "찰스 렌퍼드",
            "프랭크 다우닝",
            "아서 펜윅",
            "월터 리브스",
        ),
    }
)
# 한국식 이름으로 볼 흔한 성 (첫 글자). 이 성 + 2~3음절이면 한국 이름 풀에서만 바꾼다
KOREAN_SURNAMES = frozenset(
    "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민나진지엄채원천방공현함"
    "변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉"
)
DIGITS = re.compile(r"\d+")
SPLIT_NUMBER = re.compile(r"\d[.,·]\d")  # 소수(1.25) · 이름 속 날짜(5·18)
LETTERS = re.compile(r"[가-힣A-Za-z]")
_UNIT = re.compile(r"[가-힣A-Za-z%]{1,3}")  # 숫자 뒤 단위 (회 · 명 · 개 · mm …)
_RANGE_LINKS = ("-", "~", "–", "부터", "에서")  # 범위의 앞 덩어리 뒤에 오는 말
BOUNDED = MappingProxyType(
    {
        "month": (1, 12),
        "day": (1, 28),
        "hour": (0, 23),
        "hour12": (1, 12),  # 오전 · 오후가 붙은 시
        "minute": (1, 59),  # '12시 0분' 은 어색하다
    }
)
_DATE_KINDS = frozenset({"year", "decade", "century", *BOUNDED})
YEAR_IN_ANSWER = re.compile(r"(\d{3,4})(?=\s*년(?!대))")
QUESTION_YEAR = re.compile(r"(\d{4})\s*년")
ERA_YEAR = re.compile(
    r"(메이지|다이쇼|쇼와|헤이세이|레이와|게이오|겐지|분큐|안세이|덴포|분카|"
    r"건륭|강희|옹정|가경|도광|함풍|동치|광서|선통|광무|융희|건양|개국)\s*\d+\s*년"
)
# 연도를 다른 해에 기대어 말하는 표현 — 답의 해를 옮기면 '1981년 … 이듬해 1986년' 처럼 어긋난다.
# 해 뒤에는 조사만 허용한다 (그 해결 · 같은 해석은 아니다). 기간은 두 자리 이하만 (1985년 전반은 아니다).
# 기간(11년 동안 · 12개월 동안)도 넣는다 — '1656년부터 1667년까지 11년 동안' 의 앞 해만 옮기면 기간이 어긋난다
RELATIVE_YEAR = re.compile(
    r"(?:이듬|다음|같은|그|이전|직전)\s*해(?:에는|에|의|는|도|까지|부터|인)?(?![가-힣])|전년|"
    r"(?<!\d)\d{1,2}\s*(?:년|개월)\s*(?:뒤|후|전|만|동안|간)"
)
# 정답 연도 뒤 · 앞에 다른 연도와 범위를 잇는 말 — '2008년 2월 ~ 2009년' · '1936년에서 1950년'
RANGE_AFTER = re.compile(rf"[^.\n]{{0,{RANGE_GAP}}}?(?:~|〜|-|–|부터|에서)\s*\d{{3,4}}\s*년")
RANGE_BEFORE = re.compile(rf"\d{{3,4}}\s*년[^.\n]{{0,{RANGE_GAP}}}?(?:~|〜|-|–|부터|에서)\s*$")
ORDINAL = re.compile(r"\s*제\s*\d")  # 서수 답 (제124대)
# 같은 문장의 덧셈 · 뺄셈 서술 — '2000번대의 속번으로 1000을 더하여 3000번대' 는 수가 커도 계산값이다
ARITHMETIC = re.compile(r"더하|더한|빼면|빼고|빼서|뺀|합하|합친|합쳐")
# 범위의 앞 수와 뒤 수 사이 — '22대부터 30대까지' 의 개수(30 − 22 + 1)는 계산값이다
RANGE_BETWEEN = re.compile(r"부터|~|〜|–|-|에서")
# 연도 없는 날짜를 기념일 · 다른 일에 기대어 말하는 표현 — '5월 23일 … 이날은 서거 1주기' 는 날짜를 옮기면 거짓이 된다
ANNIVERSARY = re.compile(r"이\s*날|\d+\s*주기|\d+\s*주년|기념일|서거|탄신|생일")
MONTH_DAY = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
# 이름 뒤 원어 괄호 — 로마자 (Sotomayor) · 가나 · 한자 (足利尊氏)
ORIGINAL_SCRIPT = re.compile(r"\s*\(\s*[A-Za-z぀-ヿ㐀-鿿]")
_KOREAN_NAME = re.compile(r"[가-힣]{2,3}")
# 서양 이름 음역에만 나오는 소리 — 일본 이름은 토 · 쿠 · 루 로 적는다
_WESTERN_SYLLABLES = frozenset("트크프드브르슨")
_VOWEL_EU = 18  # 중성 번호 ㅡ — 받침 없는 ㅡ 소리(스 · 크 · 트)는 외국 이름 음역에 나오고 한국 이름엔 드물다
_JAPANESE_FINALS = (0, 4, 19)  # 일본 이름 음역의 받침 번호 — 없음 · ㄴ(겐) · ㅅ(핫토리)
# 인물 이름 근처의 성별 · 나이 단서 — 후보 이름에 성별 · 나이 정보가 없어 맞출 수 없으니 그런 문항은 뺀다
PERSON_CUE = re.compile(
    r"여공|여성|여자|여배우|여왕|그녀|부인|아내|딸|어머니|소녀|남편|아들|당시\s*\d+\s*세|\d+\s*세에|\d+\s*살"
)
# 날짜 · 숫자 정답 끝의 조사 — KorQuAD 정답에 '24일에' 처럼 붙어 온다. 떼고 바꿔야 '24일' 답도 맞는다
TRAILING_JOSA = re.compile(
    # 로 · 으로 는 넣지 않는다 — '1445유로' 의 로를 떼게 된다
    r"(?<=\d)(\s*[가-힣A-Za-z]{1,2}?)(에서|에는|에|의|은|는|이|가|을|를|부터|까지)$"
)
# 근처의 시각 — 오전 · 오후 같은 말이 앞에 있거나 뒤에 분이 붙은 것만 (시간 · 시즌 · 시장과 헷갈리지 않게)
_NEARBY_HOUR = re.compile(
    r"(?:오전|오후|밤|새벽|아침|저녁|낮)\s*(\d{1,2})\s*시|(?<!\d)(\d{1,2})\s*시\s*\d{1,2}\s*분"
)
# 받침 있는 말 뒤 · 없는 말 뒤 조사. 뒤에 한글이 이어지면 조사가 아니라 다음 낱말이라 건드리지 않는다
JOSA_PAIRS = MappingProxyType(
    {
        "이": ("이", "가"),
        "가": ("이", "가"),
        "은": ("은", "는"),
        "는": ("은", "는"),
        "을": ("을", "를"),
        "를": ("을", "를"),
        "과": ("과", "와"),
        "와": ("과", "와"),
    }
)
JOSA = r"(?:(?P<josa>으로|로|이|가|은|는|을|를|과|와)(?=[^가-힣]|$))?"


@dataclass(frozen=True)
class Passage:
    """반사실 검사가 보는 문맥 청크 하나 — 세트 행 `context` 항목의 제목과 본문."""

    title: str
    text: str

    @classmethod
    def from_json(cls, chunk: Mapping[str, Any]) -> Self:
        """세트 행 `context` 의 청크 하나로 만든다."""
        return cls(title=chunk["title"], text=chunk["text"])


@dataclass(frozen=True)
class SourceItem:
    """반사실 검사가 보는 원본 문항 — build_bench_set 이 만든 세트 행에서 검사에 쓰는 칸만."""

    item_id: str
    """세트 행의 `id` (예: `korquad:6488411-0-2`)."""
    question: str
    answers: tuple[str, ...]
    gold_title: str
    answer_shape: str
    """`strata.answer_shape` — `date_number` · `person` 등."""
    context: tuple[Passage, ...]

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """세트 행 하나로 만든다."""
        return cls(
            item_id=row["id"],
            question=row["question"],
            answers=tuple(row["answers"]),
            gold_title=row["gold_title"],
            answer_shape=row["strata"]["answer_shape"],
            context=tuple(Passage.from_json(chunk) for chunk in row["context"]),
        )


def answer_windows(text: str, pattern: re.Pattern[str], width: int = NEIGHBOR_WINDOW) -> list[str]:
    """정답 표기가 나온 곳마다 앞뒤 width 글자 (정답 표기 포함)."""
    return [
        text[max(0, match.start() - width) : match.end() + width]
        for match in pattern.finditer(text)
    ]


def context_sentences(context: Sequence[Passage], pattern: re.Pattern[str]) -> Iterator[str]:
    """청크마다 정답 표기가 나온 문장들을 차례로."""
    return chain.from_iterable(_sentences(chunk.text, pattern) for chunk in context)


def _sentences(text: str, pattern: re.Pattern[str]) -> list[str]:
    """정답 표기가 나온 문장들 — 앞뒤로 '. ' 또는 줄바꿈까지."""
    sentences = []
    for match in pattern.finditer(text):
        start = max(text.rfind(". ", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
        ends = [
            index
            for index in (text.find(". ", match.end()), text.find("\n", match.end()))
            if index >= 0
        ]
        sentences.append(text[start : min(ends, default=len(text))])
    return sentences


def name_style(name: str) -> str | None:
    """이름의 표기 계열 — "ko" 한국식(흔한 성 + 세 음절) · "ja" 일본식 · "western" 서양식 두 어절. 그 밖은 None.

    두 음절 한국 이름(성 없는 '방석')과 한 어절 · 세 어절 외국 이름은 맞출 가상 이름이 없어 None 이다.
    받침 없는 ㅡ 소리가 든 세 음절(마커스)은 성처럼 보이는 첫 글자가 있어도 외국 이름 한 어절이라 None 이다.
    """
    if (
        _KOREAN_NAME.fullmatch(name)
        and len(name) == KOREAN_NAME_LEN
        and name[0] in KOREAN_SURNAMES
        and not any(_is_open_eu(char) for char in name)
    ):
        return "ko"
    if len(name.split()) != FOREIGN_NAME_WORDS:
        return None
    return "ja" if _is_japanese_style(name) else "western"


def _is_open_eu(char: str) -> bool:
    """받침 없는 ㅡ 소리 글자인지 (스 · 크 · 트 · 즈)."""
    if not "가" <= char <= "힣":
        return False
    index = ord(char) - ord("가")
    return index % 28 == 0 and index // 28 % 21 == _VOWEL_EU


def _is_japanese_style(name: str) -> bool:
    """한글로 적은 이름이 일본 이름 음역 꼴인지 — 받침이 없거나 ㄴ · ㅅ 이고(겐타로 · 핫토리) 서양식 소리가 없다."""
    for char in name.replace(" ", ""):
        if not "가" <= char <= "힣" or char in _WESTERN_SYLLABLES:
            return False
        if (ord(char) - ord("가")) % 28 not in _JAPANESE_FINALS:
            return False
    return True


def number_type(answer: str) -> str:
    """반사실 종류 — 날짜 단위(연도 · 연대 · 세기 · 월 · 일 · 시 · 분)가 있으면 date, 아니면 number (32년 같은 기간 포함)."""
    kinds = run_kinds(answer, list(DIGITS.finditer(answer)))
    return "date" if any(kind in _DATE_KINDS for kind in kinds) else "number"


def run_kinds(value: str, runs: Sequence[re.Match[str]]) -> list[str]:
    """숫자 덩어리마다 종류. 범위(19-20세기)의 앞 덩어리는 뒤 덩어리의 종류를 따른다."""
    kinds: list[str] = []
    for index in reversed(range(len(runs))):
        run = runs[index]
        suffix = value[run.end() :].lstrip()
        if suffix.startswith(_RANGE_LINKS) and index + 1 < len(runs):
            kinds.append(kinds[-1])  # 뒤에서부터 채우므로 kinds[-1] 이 바로 뒤 덩어리
        else:
            kinds.append(_unit_kind(run.group(), suffix))
    return kinds[::-1]


def _unit_kind(run: str, suffix: str) -> str:
    """숫자 뒤 단위로 종류: year · decade · century · month · day · hour · minute · count:{단위}."""
    if suffix.startswith("년대"):
        return "decade"
    if suffix.startswith("세기"):
        return "century"
    if suffix.startswith("월"):
        return "month"
    if suffix.startswith("일"):
        return "day"
    if suffix.startswith("시"):
        return "hour"
    if suffix.startswith(("분", "초")):
        return "minute"
    if suffix.startswith("년"):
        return (
            "year" if len(run) >= YEAR_MIN_DIGITS else "count:년"
        )  # 32년 같은 두 자리 이하는 기간
    unit = _UNIT.match(suffix)
    return f"count:{unit.group() if unit else ''}"


def first_group(match: re.Match[str]) -> str:
    """여러 갈래 식에서 실제로 잡힌 첫 그룹."""
    return next(group for group in match.groups() if group is not None)


def run_pattern(kind: str, *, ordinal: bool = False) -> re.Pattern[str] | None:
    """문맥에서 같은 종류 값을 찾는 식. 일 · 분 · 단위 없는 수는 순서를 보지 않는다(None).

    ordinal 이면 '제' 가 붙은 같은 단위 서수만 찾는다 (제123대).
    """
    if kind == "month":
        return re.compile(r"(?<!\d)(\d{1,2})\s*월")
    if kind in ("hour", "hour12"):
        return _NEARBY_HOUR
    if kind == "year":
        return re.compile(r"(?<!\d)(\d{3,4})\s*년(?!대)")
    if kind == "decade":
        return re.compile(r"(?<!\d)(\d{2,4})\s*년대")
    if kind == "century":
        return re.compile(r"(?<!\d)(\d{1,2})\s*세기")
    unit = kind.removeprefix("count:")
    if kind.startswith("count:") and unit and ordinal:
        return re.compile(rf"제\s*(\d+)\s*{re.escape(unit)}")
    if kind.startswith("count:") and unit:
        return re.compile(rf"(?<![\d.])(\d+)\s*{re.escape(unit)}")
    return None


def keeps_order(values: Sequence[int], delta: int, neighbors: Sequence[int]) -> bool:
    """옮긴 값이 근처 같은 종류 값들과의 앞뒤 순서를 지키고 겹치지 않는지."""
    return all(
        (value - neighbor > 0) == (value + delta - neighbor > 0) and value + delta != neighbor
        for value, neighbor in product(values, neighbors)
    )


ROLE_SUFFIXES = (
    "감독",
    "회장",
    "장관",
    "위원",
    "대통령",
    "의원",
    "시장",
    "총재",
    "교수",
    "선수",
    "사장",
    "장군",
    "천황",
    "목사",
    "코치",
    "원장",
    "총장",
    "총리",
    "부인",
)
# 사람이 아닌 답의 꼬리 — answer_shape person 에 섞여 들어온 장소 · 무리 · 개념 (베르사유 궁전 · 누리꾼 · 콩도르세 패자)
_NOT_PERSON_SUFFIXES = ("궁전", "꾼", "패자")
# 사람을 묻는 질문 — 이 말이 없으면(1976년 사고 당시 차종은? → 페라리) 인물 문항으로 치지 않는다
PERSON_QUESTION = re.compile(
    r"누구|누가|사람|인물|"
    r"(선수|평론가|설립자|작가|화가|학자|장군|대통령|총리|의원|감독)[은는이가]?\s*\?\s*$"  # '… 지시한 선수는?'
)


def is_name(answer: str) -> bool:
    """사람 이름꼴 답인지 — 숫자 · 직함 꼬리 · 긴 구절을 걸러 반사실 인물 치환에 쓸 만한지 본다."""
    return (
        not any(char.isdigit() for char in answer)
        and len(answer) <= NAME_MAX_LEN
        and not answer.endswith(ROLE_SUFFIXES)
        and not answer.endswith(_NOT_PERSON_SUFFIXES)
    )


def flexible_pattern(text: str) -> re.Pattern[str]:
    """글자 사이 공백을 허용하는 식. 숫자로 시작 · 끝나면 앞뒤 숫자 경계를 둔다 — '5명' 이 '15명' 속에서 잡히지 않게."""
    chars = [re.escape(char) for char in text if not char.isspace()]
    stripped = text.strip()
    prefix = r"(?<!\d)" if stripped[:1].isdigit() else ""
    suffix = r"(?!\d)" if stripped[-1:].isdigit() else ""
    return re.compile(prefix + r"\s*".join(chars) + suffix)
