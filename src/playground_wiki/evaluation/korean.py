"""한글 음절의 받침 계산 — 조사 고르기 두 곳(wikidata 질문 틀 · 반사실 치환)이 같이 쓴다."""

_FIRST_SYLLABLE = "가"
_LAST_SYLLABLE = "힣"
_FINALS = 28  # 음절마다 받침 자리 수 (받침 없음 포함)
RIEUL_FINAL = 8
"""ㄹ 받침 번호 — '로' 를 받침 없는 말처럼 받는다."""


def final_consonant(char: str) -> int | None:
    """한글 음절의 받침 번호 (0 이면 받침 없음). 한글 음절이 아니면 None."""
    if not _FIRST_SYLLABLE <= char <= _LAST_SYLLABLE:
        return None
    return (ord(char) - ord(_FIRST_SYLLABLE)) % _FINALS
