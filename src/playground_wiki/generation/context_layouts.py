"""위키의 컨텍스트 칸 정의 (초안): 단발 `/ask` 와 에이전트 노드의 칸 배치 · 칸 예산 · 후보 바꾸기.

답변 경로에는 아직 잇지 않았다 — 칸 예산 숫자는 연결할 때 응답의 프롬프트 토큰 수로 다시 정한다 (context-design §11).
전체 예산은 `num_ctx` 에서 답 길이를 뺀 값으로 쓰는 쪽(api)이 넘긴다 — 설정 파일은 api 만 읽는다.
"""

from collections.abc import Sequence

from playground_retrieval.core import Hit

from playground_wiki.generation.context import Candidate, ContextLayout, Evidence, Slot

SYSTEM = "system"
"""고정 지시 (base + 단계 블록). system 턴."""
DOCUMENTS = "documents"
"""검색 문서 · 모은 근거. `<document id="n" title="…">` — 답 속 `[n]` 과 같은 번호 (ADR-026)."""
QUESTION = "question"
"""사용자 질문 원문."""
SUB_QUESTIONS = "sub_questions"
"""분해 노드가 만든 하위 질문 (에이전트가 쓴 글 — data)."""
QUERIES = "queries"
"""지금까지 쓴 검색어 · 새 검색어 (에이전트가 쓴 글 — data)."""
NOTES = "notes"
"""지난 반복에서 찾은 사실 메모 (에이전트가 쓴 글 — data)."""
NODE = "node"
"""노드별 지시 (분해 · 판단 · 새 검색어 · 답변). 우리가 쓴 글이라 user 턴에 태그 없이."""
REMINDER = "reminder"
"""질문 한 줄 상기 — 근거 뒤에 질문을 다시 둬 위치 효과를 챙긴다 (context-design §5)."""

DOCUMENT_TAG = "document"
DOCUMENTS_TAG = "documents"
"""`question@v1` 틀이 문서 블록 밖을 감싸는 태그. 단발에서는 틀이 쓰므로 흉내 지우기에만 넣는다."""
QUESTION_TAG = "question"

# 칸 예산 (토큰, 글자 수 어림 2/3). 초안 — 연결 때 응답의 prompt 토큰 수로 확인해 다시 정한다
SYSTEM_BUDGET = 4500  # base@v4 + grounded@v5 ≈ 5,900자 ≈ 3,900토큰 + 다음 판 여유
DOCUMENTS_BUDGET = 2400  # 지금 단발: 청크 5 × 384토큰 + 태그 · 제목
QUESTION_BUDGET = 300  # 질문 300자 상한 ≈ 200토큰 + 태그
SUB_QUESTIONS_BUDGET = 300  # 하위 질문 서너 개
QUERIES_BUDGET = 300  # 검색어 최대 3회 × 하위 질문
NOTES_BUDGET = 600  # 반복마다 한 줄씩, 최대 3회
NODE_BUDGET = 800  # 노드별 지시 문구 (아직 없음)
AGENT_DOCUMENTS_BUDGET = 3000  # 하위 질문마다 근거가 쌓여 단발보다 넓게


def ask_layout(budget: int) -> ContextLayout:
    """단발 `/ask` 의 칸: system(지시) → user(문서 · 질문).

    documents 칸 글은 지금 `format_context` 와 같다 — 연결 때 `question@v1` 틀의 `$context` 에 넣으면
    같은 프롬프트 · 같은 prompt_version 이다. question 칸은 틀이 질문을 붙이는 동안 예산을 잡아 두는 자리.
    """
    return ContextLayout(
        slots=(
            Slot(SYSTEM, "instruction", budget=SYSTEM_BUDGET, role="system"),
            Slot(
                DOCUMENTS,
                "data",
                budget=DOCUMENTS_BUDGET,
                priority=1,
                item_tag=DOCUMENT_TAG,
                numbered=True,
            ),
            Slot(QUESTION, "data", budget=QUESTION_BUDGET, tag=QUESTION_TAG, required=True),
        ),
        budget=budget,
        reserved_tags=frozenset({DOCUMENTS_TAG}),
    )


def agent_layout(budget: int) -> ContextLayout:
    """에이전트 노드 공통 칸 (context-design §5): 고정 지시 → 질문 → 중간 산출물 · 근거 → 노드별 지시 → 질문 상기.

    모든 노드가 같은 배치를 쓰고 노드별 지시만 바꾼다 — 앞부분(고정 지시 · 질문)을 노드끼리 공유한다.
    근거 · 메모는 받은 순서(given)로 놓아 반복마다 뒤에 붙는다.
    """
    return ContextLayout(
        slots=(
            Slot(SYSTEM, "instruction", budget=SYSTEM_BUDGET, role="system"),
            Slot(QUESTION, "data", budget=QUESTION_BUDGET, tag=QUESTION_TAG, required=True),
            Slot(
                SUB_QUESTIONS,
                "data",
                budget=SUB_QUESTIONS_BUDGET,
                priority=1,
                tag="sub_questions",
                item_tag="sub_question",
                order="given",
            ),
            Slot(
                DOCUMENTS,
                "data",
                budget=AGENT_DOCUMENTS_BUDGET,
                priority=2,
                tag=DOCUMENTS_TAG,
                item_tag=DOCUMENT_TAG,
                numbered=True,
                order="given",
            ),
            Slot(
                QUERIES,
                "data",
                budget=QUERIES_BUDGET,
                priority=3,
                tag="queries",
                item_tag="query",
                order="given",
            ),
            Slot(
                NOTES,
                "data",
                budget=NOTES_BUDGET,
                priority=3,
                tag="notes",
                item_tag="note",
                order="given",
            ),
            Slot(NODE, "instruction", budget=NODE_BUDGET),
            Slot(REMINDER, "data", budget=QUESTION_BUDGET, tag=QUESTION_TAG, required=True),
        ),
        budget=budget,
    )


def hit_candidates(hits: Sequence[Hit], *, slot: str = DOCUMENTS) -> list[Candidate]:
    """검색 결과를 문서 칸 후보로 바꾼 것을 돌려준다 (key 는 청크 id, 묶음은 문서 id, 속성은 제목)."""
    return [
        Candidate(
            slot=slot,
            text=hit.chunk.text,
            score=hit.score,
            key=hit.chunk.id,
            group=hit.chunk.doc_id,
            attributes={"title": hit.chunk.title},
        )
        for hit in hits
    ]


def hit_evidence(hits: Sequence[Hit], *, query: str) -> list[Evidence]:
    """검색 결과를 작업 기억에 쌓을 근거로 바꾼 것을 돌려준다."""
    return [
        Evidence(
            key=hit.chunk.id,
            text=hit.chunk.text,
            score=hit.score,
            query=query,
            group=hit.chunk.doc_id,
            attributes={"title": hit.chunk.title},
        )
        for hit in hits
    ]


class NoMemory:
    """위키의 기억 조회 자리 — 늘 빈 후보 (`CandidateSource` 의 빈 구현).

    위키는 단발 · 로그인 없음 · 대화 기록 없음 전제를 지킨다. 장기 메모리는 소설 프로젝트에서 이 자리에 끼운다.
    """

    async def collect(self, query: str, budget: int) -> list[Candidate]:
        """빈 목록을 돌려준다."""
        del query, budget  # 기억이 없어 보지 않는다
        return []
