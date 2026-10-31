"""질문 관문: 잡담 · 실시간 · 자해 질문을 검색 전에 어휘 규칙으로 골라 고정 답을 준다 (ADR-025).

위키 밖 질문을 거절하는 일은 임계값이 아니라 partial 프롬프트의 "알 수 없음"과 이 관문이 나눠 맡는다.
규칙은 좁게 둔다 — 놓친 질문은 검색 · partial 로 가서 "알 수 없음"이 받지만, 잘못 걸린 질문은 답을 잃는다.

평가: 관문의 지표는 정밀도다. 재현율은 검색 · LLM 을 건너뛰어 아끼는 지연의 크기일 뿐이다.
규칙을 모르는 쪽이 쓴 세트 tests/gate/gate-blind-v1.jsonl (2026-10-24, 실시간 50 · 헷갈리는 위키 질문 50)로
실시간 14/50 · 위키 오탐 0/50. 문항을 보고 만든 offwiki 실시간으로는 12/12 였다 — 같은 세트로 고치고 재면 숫자가 부푼다.
목록을 고칠 때는 이 세트(1)를 보고 고치고, 규칙을 모르는 쪽이 새로 쓴 세트(2)로 다시 잰다. 재현율 숫자만 보고 목록을
불리지 않는다 — 넣을 낱말은 놓친 질문이 검색 · partial 에서 옛 값을 지금 값처럼 답한 꼴만이다.
놓친 36개를 검색 → 답변 정책 → 생성(gemma4:12b, base@v2)까지 넣었더니 35개는 거절 · "알 수 없음" · 날짜 붙은 옛 값으로
답했고, 1개("비트코인 시세")만 문서의 '현재'를 옮겨 옛 값을 지금 값처럼 답했다 — 그 꼴을 PRICE_QUOTES 로 넣었다.
그래서 세트 1 의 실시간 15/50 은 세트를 보고 고친 뒤의 숫자라 재현율로 쓰지 않는다.

자해 규칙(safety)만 이 평가 기준이 다르다. 실시간 규칙은 놓쳐도 뒤에서 "알 수 없음"이 받지만, 자해 · 자살 방법 질문은
받아 줄 뒤 단계가 없다 — 유해 질문 세트(2026-10-29)에서 partial 이 문서의 방법 서술을 견주어 답한 것을 사람이 해롭다고
봤고, 같은 질문에 상담 안내를 붙이면서도 그렇게 했다. 그래서 이 갈래만 LLM 앞에서 끊는다. 재현율은 여전히 바닥이다
(완곡한 표현은 어휘로 잡히지 않는다) — 이 규칙은 커버리지가 아니라 바닥이고, 나머지는 partial@v3 이 받는다.
오탐이 나도 사용자가 받는 것은 상담 안내라 잃는 답이 없다는 점이 실시간 규칙과 다르다.

그루밍 규칙(safety)도 같은 이유로 LLM 앞에서 끊는다 (2026-10-31). 아동 대상 그루밍 방법 질문(유해 세트 hz-14)은 partial 로
바로 가는데, partial@v4 가 막는지가 앞선 요청 이력에 따른 생성 흔들림으로 갈렸다 — 같은 프롬프트 · 같은 문맥에서 3회 중
0 번(v4 측정)과 2 번(v5 측정)이 사람 판정 해로웠다. 어휘(대상 말 · 방법을 묻는 말 · 제외 맥락)와 고정 답은 사용자가 정했다.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, get_args

from playground_wiki.gate.tokens import Token, Tokenizer

Rule = Literal["smalltalk", "realtime", "safety"]
Kind = Literal[
    "greeting", "thanks", "farewell", "identity", "chitchat", "realtime", "crisis", "grooming"
]
KINDS: tuple[str, ...] = get_args(Kind)
RULE_BY_KIND: Mapping[str, Rule] = {
    "realtime": "realtime",
    "crisis": "safety",
    "grooming": "safety",
}
"""잡담 말고 자기 규칙을 가진 종류 → 규칙 이름. 여기 없는 종류는 모두 잡담 규칙이다."""

# 잡담 사전 — 질문 전체가 정규화한 뒤 이 중 하나와 같아야 걸린다. 섞인 문장("안녕, 세종대왕은 누구야?")은 검색으로 간다.
# 정규화가 공백 · 문장부호 · 이모지 · ㅋㅋ · ㅠㅠ · ~ 를 지우고 소문자로 바꾸니 그 변형은 따로 적지 않는다.
# 위키 문서 제목이기도 한 낱말(감사 · 대박 · 수고)은 넣지 않는다
SMALLTALK: Mapping[str, tuple[str, ...]] = {
    "greeting": (
        "안녕",
        "안녕하세요",
        "안녕하십니까",
        "하이",
        "ㅎㅇ",
        "hi",
        "hello",
        "hey",
        "헬로",
        "반가워",
        "반가워요",
        "반갑습니다",
        "처음 뵙겠습니다",
        "좋은 아침",
        "굿모닝",
    ),
    "thanks": (
        "고마워",
        "고마워요",
        "고맙습니다",
        "고맙다",
        "감사해",
        "감사해요",
        "감사합니다",
        "ㄱㅅ",
        "ㄳ",
        "땡큐",
        "thanks",
        "thank you",
        "thx",
        "덕분이야",
        "도움 됐어",
    ),
    "farewell": (
        "잘 가",
        "잘 가요",
        "안녕히 가세요",
        "안녕히 계세요",
        "바이",
        "바이바이",
        "빠이",
        "bye",
        "잘 자",
        "잘 자요",
        "굿나잇",
        "또 봐",
        "다음에 봐",
        "수고했어",
        "수고하세요",
    ),
    "identity": (
        "너 누구야",
        "넌 누구야",
        "너는 누구야",
        "누구세요",
        "너 뭐야",
        "넌 뭐야",
        "너는 뭐야",
        "너 이름이 뭐야",
        "이름이 뭐야",
        "이름 뭐야",
        "네 이름은",
        "자기소개 해 줘",
        "너 뭐 할 수 있어",
        "뭐 할 수 있어",
        "넌 뭘 할 수 있어",
    ),
    "chitchat": (
        "심심해",
        "뭐 해",
        "뭐 하고 있어",
        "기분 어때",
        "오늘 기분 어때",
        "오늘 하루 어땠어",
        "배고파",
        "졸려",
        "피곤해",
        "그래서",
        "그래",
        "응",
        "ㅇㅇ",
        "아니",
        "네",
        "넵",
        "ok",
        "오케이",
        "알겠어",
        "좋아",
    ),
}
# 웃음 · 울음 자모 (ㅋ · ㅠ · ㅜ 는 하나라도, ㅎ 는 'ㅎㅇ' 인사가 있어 두 개 이상만).
# NFKC 가 'ㅋㅠ' 같은 호환 자모를 음절('큐')로 합칠 수 있어 NFKC 전에 지운다. 첫소리 · 가운뎃소리 자모로 들어온 것도 함께
# 호환 자모 ㅋ ㅠ ㅜ ㅎ = U+314B U+3160 U+315C U+314E, 첫소리 ㅋ ㅎ = U+110F U+1112, 가운뎃소리 ㅠ ㅜ = U+1172 U+116E
_LAUGH_CRY = re.compile("[\u314b\u3160\u315c\u110f\u1172\u116e]+|[\u314e\u1112]{2,}")
# 지우는 유니코드 범주: 문장부호(P) · 기호(S, 이모지 · ~ 포함) · 공백(Z) · 제어 · 서식(C, ZWJ) · 결합 표시(Mn, 이모지 변형 선택자)
_DROP_CATEGORIES = ("P", "S", "Z", "C", "Mn")

# 실시간 규칙 — 때를 가리키는 말과 바뀌는 값을 묻는 말이 한 질문에 함께 있어야 걸린다.
# 형태소 형태의 연속으로 맞춘다. '현재' 는 KorQuAD dev 5,774 문항에 27번 나오지만 대상어와 함께 나온 적은 없다.
# 받아들인 오탐: "현재 가격 혁명 시기의 유럽 경제" (현재/MAG + 가격). '요즘' · '올해' · '최근' 은 덤프에 답이 있을 수 있어 넣지 않는다
TIME_MARKERS: tuple[tuple[str, ...], ...] = (
    ("지금",),
    ("오늘",),
    ("현재",),
    ("실시간",),
    ("방금",),
    ("어제",),
    ("내일",),
    ("이번", "주"),
    ("이번", "주말"),
)
# 때 표지 바로 뒤에 이 조사가 오면 누적 · 시작점이지 지금 값이 아니다 (지금까지 가장 많이 팔린 · 오늘부터).
# 다른 조사는 막지 않는다 — "오늘은 날씨 어때?" · "지금은 몇 시야?" · "오늘의 날씨" 는 오늘/NNG + 조사로 나온다
RANGE_PARTICLES = frozenset({"까지", "부터"})
# 때 표지 바로 뒤에 접미사가 오면 다른 낱말이다 (현재형 = 현재/NNG + 형/XSN).
# 어림 접미사는 예외 — "지금쯤 택배 어디야" 도 지금/NNG + 쯤/XSN 으로 나온다
SUFFIX_TAG = "XSN"
APPROX_SUFFIXES = frozenset({"쯤", "경", "께"})
# 위키로 답하는 질문에 걸려 넣지 않는 것: 지수(인간개발지수 · 물가 지수 · 지수함수) · 순위("현재 세계 인구 순위") ·
# 뉴스("현재 KBS 뉴스 9 앵커는 누구" — "오늘 뉴스 헤드라인" 은 헤드라인으로 잡는다).
# Kiwi 가 쪼개는 낱말은 이어진 형태로 적는다: 미세먼지 = 미세 + 먼지, 시청률 = 시청 + 률, 몇 시 = 몇 + 시 (몇 시간과 다르다)
TARGETS: tuple[tuple[str, ...], ...] = (
    ("날씨",),
    ("기온",),
    ("미세", "먼지"),
    ("황사",),
    ("환율",),
    ("주가",),
    ("코스피",),
    ("코스닥",),
    ("나스닥",),
    ("비트코인",),
    ("코인",),
    ("시세",),
    ("가격",),
    ("로또",),
    ("당첨",),
    ("헤드라인",),
    ("속보",),
    ("며칠",),
    ("요일",),
    ("날짜",),
    ("몇", "시"),
    ("출발",),
    ("도착",),
    ("택배",),
    ("배송",),
    ("경기", "결과"),
    ("스코어",),
    ("시청", "률"),
)
# 때 표지 없이도 걸리는 꼴 — 자산 바로 뒤의 '시세' 는 그 자체로 지금 값을 묻는다 ("비트코인 시세").
# **'때 표지 + 대상어' 두 조건 설계의 유일한 예외다.** 규칙 모르는 세트에서 "비트코인 시세" 가 관문을 지나 grounded 로
# 가서 문서의 '현재'를 옮겨 옛 값을 지금 값처럼 답했다 — 검색 → 답변 정책 → 생성을 끝까지 돌려 해로운 답을 본 증거로 넣었다.
# 단일 조건 규칙을 더 넣으려면 놓친 질문이 있다는 것만으로는 부족하고, 같은 급의 끝까지 돌린 증거가 있어야 한다.
# 시세 뒤에 명사가 오면(시세 차익 · 시세조종) 개념 질문, 연도(1997년 금 시세)나 과거 표지(작년 · 역대 · 초기 …)가 있으면
# 옛 값을 묻는 것이라 걸지 않는다. '시세' 하나로는 걸지 않는다("조선 후기 쌀 시세"). 자산은 닫힌 목록 — 종목명까지 열면 끝이 없다
PRICE_QUOTES: tuple[tuple[str, ...], ...] = tuple(
    (asset, "시세") for asset in ("비트코인", "이더리움", "코인", "금", "달러", "엔화", "유로")
)
NOUN_TAGS = frozenset({"NNG", "NNP", "NNB", "XSN"})
# 숫자 없이 과거를 가리키는 말 (한 낱말로 나오는 것)
PAST_MARKERS = frozenset(
    {"작년", "지난해", "지난달", "지난주", "역대", "당시", "과거", "예전", "초기"}
)
# 띄어 쓴 '지난 해 · 달 · 주' 는 지나/VV + ㄴ/ETM(U+11AB 받침 자모) + 해 · 달 · 주 세 토큰으로 나온다 — 비표준이지만 흔하다
PAST_PHRASES: tuple[tuple[str, ...], ...] = tuple(
    ("지나", "\u11ab", unit) for unit in ("해", "달", "주")
)

# 자해 규칙 — 실시간 규칙과 같은 두 조건 꼴이다: 자해 · 자살을 가리키는 말과 방법을 직접 묻는 말이 한 질문에 함께 있어야
# 걸린다. 좁게 둔다 (모듈 docstring 의 '자해 규칙' 문단).
SELF_HARM: tuple[tuple[str, ...], ...] = (("자살",), ("자해",))
# 방법을 직접 묻는 말. '법' 하나는 넣지 않는다 — "자살 관련 법은?" 같은 법률 질문이 걸린다.
# '어떻게'는 때에 따라 어떻/VA-I + 게/EC 로 쪼개져 이 형태와 다르다 ("자살률은 어떻게 변했나") — 부사일 때만 맞는다
METHOD_ASKS: tuple[tuple[str, ...], ...] = (("방법",), ("수단",), ("어떻게",))
# 이 말이 하나라도 있으면 걸지 않는다 — 예방 · 통계 · 제도 · 보도처럼 위키백과가 답해야 하는 질문이다.
# Kiwi 가 쪼개는 꼴에 맞춰 적는다: '자살예방법' = 자살 + 예방법, '자살률' = 자살 + 률(XSN), '상담사'는 한 낱말
SAFE_CONTEXT = frozenset(
    {
        "예방",
        "예방법",
        "상담",
        "상담사",
        "통계",
        "률",
        "정책",
        "제도",
        "신고",
        "처벌",
        "법률",
        "보도",
        "유가족",
        "추모",
        "교육",
        "캠페인",
        "원인",
        "실태",
    }
)
# 그루밍 규칙 — 사용자가 정한 어휘 (2026-10-31). 대상 말과 방법을 묻는 말이 함께 있고 제외 맥락이 없어야 걸린다.
# 대상 말은 형태소가 아니라 정규화한 질문(normalize)에서 찾는다 — Kiwi 가 '그루머'를 그 + 루머로 쪼개고, 띄어 쓴
# '성적 그루밍 · 아동 그루밍 · 그루밍 범죄' 등은 모두 '그루밍'을 품는다
GROOMING_TARGETS: tuple[str, ...] = ("그루밍", "그루머", "grooming", "groomer")
# Kiwi 가 쪼개는 꼴: '어떻게'는 부사(어떻게)와 어떻/VA-I + 게/EC 둘 다, '하는 법' = 하 + 는 + 법, '하려면' = 하 + 려면
GROOMING_ASKS: tuple[tuple[str, ...], ...] = (
    ("방법",),
    ("수단",),
    ("어떻게",),
    ("어떻", "게"),
    ("절차",),
    ("단계",),
    ("순서",),
    ("과정",),
    ("수법",),
    ("요령",),
    ("기술",),
    ("팁",),
    ("노하우",),
    ("매뉴얼",),
    ("하", "는", "법"),
    ("하", "려면"),
)
# 제외 맥락 — 이 형태가 하나라도 있으면 걸지 않는다. '음모론' = 음모 + 론,
# '알아차리다'는 활용형에서 알아차리 한 토큰. '법'은 '하는 법'의 법(앞이 관형형 어미)이면 제외로 보지 않는다.
# 뜻풀이 말(뜻 · 무엇 · 뭐 · ~이란)과 '피해'는 넣지 않는다 — 규칙은 방법 말이 있어야 걸려 순수한 뜻 질문은 어차피
# 통과하고, 넣으면 '수법이 뭐야 · 피해를 주는 방법' 같은 방법 질문이 빠져나간다. 예방 · 방지도 같은 이유로 넣지 않는다
GROOMING_SAFE_CONTEXT = frozenset(
    {
        "징후",
        "신호",
        "알아차리",
        "의심",
        "신고",
        "처벌",
        "법",
        "법률",
        "형량",
        "판결",
        "재판",
        "사건",
        "뉴스",
        "기사",
        "피해자",
        "상담",
        "치료",
        "회복",
        "통계",
        "실태",
        "역사",
        "유래",
        "음모",
        "논란",
        "강아지",
        "고양이",
        "반려",
        "애견",
        "미용",
        "털",
        "셀프",
        "화장품",
    }
)
CRISIS_LINE = "109"
"""자살예방 상담전화 번호 — 2024년에 109 로 통합됐다. 바뀌면 이 상수와 partial 프롬프트의 안내를 같이 고친다
(tests/gate/test_gate.py 가 두 곳이 같은 번호인지 본다). 해마다 한 번 맞는지 확인한다."""

# 종류별 고정 답. 실시간 답의 기준일은 덤프 날짜(configs/ingest.yaml source.dump_date)를 넣는다
REPLIES: Mapping[str, str] = {
    "greeting": "안녕하세요! 한국어 위키백과에 있는 내용을 물어보세요.",
    "thanks": "도움이 됐다니 다행이에요. 더 궁금한 게 있으면 물어보세요.",
    "farewell": "안녕히 가세요. 궁금한 게 생기면 또 물어보세요.",
    "identity": (
        "저는 한국어 위키백과 문서를 찾아 답하는 도우미예요. "
        "사람 · 사건 · 개념처럼 위키백과에 있는 내용을 물어보세요."
    ),
    "chitchat": (
        "저는 위키백과 질문에 답하는 도우미라 잡담은 잘 못해요. 궁금한 사람 · 사건 · 개념을 물어보세요."
    ),
    "realtime": (
        "날씨 · 시세 · 뉴스 · 시각 같은 실시간 정보는 알 수 없어요. "
        "저는 {dump_date} 기준 한국어 위키백과 문서로만 답합니다."
    ),
    "crisis": (
        f"그 방법은 알려 드릴 수 없어요. 지금 많이 힘드시다면 자살예방 상담전화 {CRISIS_LINE}번에서 "
        "24시간 상담을 받을 수 있어요. 전문 상담원과 바로 이야기할 수 있습니다."
    ),
    # 사용자가 정한 문구 (2026-10-31). 안내한 질문(뜻 · 징후 · 신고 방법)은 이 규칙에 다시 걸리지 않는다 — '예방'은 제외
    # 맥락에 없어 안내하지 않는다. 112 · 여성긴급전화 1366 — 바뀌면 이 문구를 고친다
    "grooming": (
        "아동을 대상으로 한 그루밍(성적 길들이기) 수법이나 절차는 알려 드리지 않습니다. "
        "그루밍의 뜻이나 징후, 신고 방법이 궁금하시면 그렇게 물어봐 주세요. "
        "아동 대상 성범죄가 의심되면 112 또는 여성긴급전화 1366으로 연락할 수 있습니다."
    ),
}


@dataclass(frozen=True)
class GateMatch:
    """관문에 걸린 질문. 어떤 규칙 · 토큰이 걸었는지 로그 · 응답 메타데이터에 남긴다."""

    kind: Kind
    matched: tuple[str, ...]
    """건 근거 — 잡담은 정규화한 질문(정규화 뒤 빈 질문이면 비어 있다), 실시간은 때 표지와 대상어,
    자해는 자해 · 자살을 가리킨 말과 방법을 묻는 말."""

    @property
    def rule(self) -> Rule:
        """걸린 규칙."""
        return RULE_BY_KIND.get(self.kind, "smalltalk")


def normalize(text: str) -> str:
    """잡담 사전과 맞출 꼴. 웃음 · 울음 자모를 지우고 NFKC · 소문자로 바꾼 뒤 공백 · 문장부호 · 기호를 지운다."""
    text = unicodedata.normalize("NFKC", _LAUGH_CRY.sub("", text)).lower()
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(_DROP_CATEGORIES))


def smalltalk_table(lexicon: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """잡담 사전을 정규화한 질문 → 종류 표로. 정규화 뒤 비거나 겹치는 말이 있으면 실수라 막는다."""
    table: dict[str, str] = {}
    for kind, phrases in lexicon.items():
        if kind not in KINDS:
            raise ValueError(f"unknown smalltalk kind {kind!r}")
        for phrase in phrases:
            key = normalize(phrase)
            if not key or key in table:
                raise ValueError(
                    f"smalltalk phrase {phrase!r} is empty or duplicated after normalize"
                )
            table[key] = kind
    return table


def realtime_match(tokens: Sequence[Token]) -> GateMatch | None:
    """때 표지와 대상어가 함께 있거나 '자산 시세' 꼴이면 실시간 질문. 근거는 나온 때 표지 · 대상어 전부."""
    forms = [token.form for token in tokens]
    markers = [
        pattern
        for end, pattern in _find(forms, TIME_MARKERS)
        if not _part_of_other_word(tokens, end)
    ]
    targets = [pattern for _, pattern in _find(forms, TARGETS)]
    if markers and targets:
        return GateMatch("realtime", tuple(" ".join(pattern) for pattern in markers + targets))
    quotes = [
        pattern
        for end, pattern in _find(forms, PRICE_QUOTES)
        if not (end < len(tokens) and tokens[end].tag in NOUN_TAGS)
    ]
    if quotes and not _points_to_past(tokens):
        return GateMatch("realtime", tuple(" ".join(pattern) for pattern in quotes))
    return None


def crisis_match(tokens: Sequence[Token]) -> GateMatch | None:
    """자해 · 자살을 가리키는 말과 방법을 묻는 말이 함께 있으면 위기 질문. 근거는 나온 말 전부.

    예방 · 통계 · 제도처럼 위키백과가 답할 질문을 가리키는 말(SAFE_CONTEXT)이 하나라도 있으면 걸지 않는다.
    """
    forms = [token.form for token in tokens]
    if any(form in SAFE_CONTEXT for form in forms):
        return None
    harm = [pattern for _, pattern in _find(forms, SELF_HARM)]
    asks = [pattern for _, pattern in _find(forms, METHOD_ASKS)]
    if harm and asks:
        return GateMatch("crisis", tuple(" ".join(pattern) for pattern in harm + asks))
    return None


def fixed_reply(match: GateMatch, *, dump_date: str) -> str:
    """걸린 종류의 고정 답. dump_date 는 사람이 읽는 꼴('2026년 10월 1일')."""
    return REPLIES[match.kind].format(dump_date=dump_date)


def grooming_match(question: str, tokens: Sequence[Token]) -> GateMatch | None:
    """그루밍을 가리키는 말과 방법을 묻는 말이 함께 있으면 그루밍 방법 질문. 근거는 나온 말 전부.

    뜻 · 징후 · 신고 · 처벌처럼 위키백과가 답할 질문이나 반려동물 · 미용의 그루밍을 가리키는 말(GROOMING_SAFE_CONTEXT)이
    하나라도 있으면 걸지 않는다.
    """
    key = normalize(question)
    targets = [target for target in GROOMING_TARGETS if target in key]
    if not targets:
        return None
    forms = [token.form for token in tokens]
    for index, token in enumerate(tokens):
        if token.form not in GROOMING_SAFE_CONTEXT:
            continue
        # '하는 법'의 법은 방법을 묻는 말이다 — 바로 앞이 관형형 어미면 제외 맥락으로 보지 않는다
        if token.form == "법" and index > 0 and tokens[index - 1].tag == "ETM":
            continue
        return None
    asks = [pattern for _, pattern in _find(forms, GROOMING_ASKS)]
    if asks:
        return GateMatch("grooming", tuple(targets) + tuple(" ".join(pattern) for pattern in asks))
    return None


def _find(
    forms: Sequence[str], patterns: Sequence[tuple[str, ...]]
) -> list[tuple[int, tuple[str, ...]]]:
    """형태 열에서 패턴이 나온 (끝 위치 — 다음 토큰 자리, 패턴). 앞에서부터.

    여러 형태소 패턴(시청 + 률)은 한 토큰(시청률)으로 나와도 맞는다 — 사용자 사전에 낱말이 등록되면 경계가 합쳐진다.
    """
    found = []
    for start in range(len(forms)):
        for pattern in patterns:
            if tuple(forms[start : start + len(pattern)]) == pattern:
                found.append((start + len(pattern), pattern))
            elif len(pattern) > 1 and forms[start] == "".join(pattern):
                found.append((start + 1, pattern))
    return found


def _points_to_past(tokens: Sequence[Token]) -> bool:
    """'1997년' 같은 연도나 과거 표지(작년 · 지난 해 · 역대 · 초기 …)가 있으면 옛 값을 묻는 질문이다."""
    has_year = any(token.tag == "SN" and after.form == "년" for token, after in pairwise(tokens))
    past = any(token.form in PAST_MARKERS for token in tokens)
    return has_year or past or bool(_find([token.form for token in tokens], PAST_PHRASES))


def _part_of_other_word(tokens: Sequence[Token], end: int) -> bool:
    """때 표지 바로 뒤 토큰이 누적 · 시작점 조사(까지 · 부터)나 접미사(어림 접미사 말고)면 지금을 가리키는 말이 아니다."""
    if end >= len(tokens):
        return False
    after = tokens[end]
    if after.tag == SUFFIX_TAG:
        return after.form not in APPROX_SUFFIXES
    return after.form in RANGE_PARTICLES and after.tag.startswith("J")


class QueryGate:
    """질문을 검색 전에 본다. 잡담이면 종류별, 실시간이면 덤프 날짜를 넣은, 자해면 상담 안내, 그루밍 방법이면 신고처 안내 고정 답으로 끝낸다.

    예)
        gate = QueryGate(KiwiTokenizer())
        match = gate.check(question)
        if match is not None:
            logger.info("gate %s %s", match.kind, match.matched)
            return fixed_reply(match, dump_date="2026년 10월 1일")
    """

    def __init__(
        self, tokenizer: Tokenizer, lexicon: Mapping[str, Sequence[str]] = SMALLTALK
    ) -> None:
        self._tokenizer = tokenizer
        self._smalltalk = smalltalk_table(lexicon)

    def check(self, question: str) -> GateMatch | None:
        """걸리면 근거와 함께, 아니면 None (검색 → 답변 정책으로). 정규화 뒤 빈 질문은 잡담으로 본다."""
        key = normalize(question)
        if not key:
            return GateMatch("chitchat", ())
        if key in self._smalltalk:
            return GateMatch(self._smalltalk[key], (key,))
        tokens = self._tokenizer.tokenize(question)
        return crisis_match(tokens) or grooming_match(question, tokens) or realtime_match(tokens)
