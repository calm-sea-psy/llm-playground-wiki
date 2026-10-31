"""위키의 컨텍스트 칸 정의: 단발 `/ask` 와 에이전트 노드의 칸 배치 · 칸 예산 · 후보 바꾸기.

단발 `/ask` 의 문서 칸은 `ask_documents` 가 만든다 (`build_prompt` 가 부른다). 칸 예산 숫자는 글자 수 어림이라
응답의 프롬프트 토큰 수로 다시 확인한다 (context-design §11). 전체 예산은 쓰는 쪽이 프로필의 `num_ctx` 로 넘긴다.
"""

from collections.abc import Sequence

from playground_llm import Message
from playground_retrieval.core import Hit

from playground_wiki.generation.context import (
    BudgetError,
    Candidate,
    CharTokenCounter,
    ContextAssembler,
    ContextLayout,
    Evidence,
    Slot,
)

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

# 칸 예산 (토큰, 글자 수 어림 2/3). 응답의 prompt 토큰 수로 확인해 다시 정한다
ASK_BUDGET = 8192  # 단발 전체 상한 — 프로필에 num_ctx 가 없을 때(시험 프로필). 운영은 answer.yaml 의 num_ctx 를 넘긴다
SYSTEM_BUDGET = 4500  # base@v4 + grounded@v5 ≈ 5,900자 ≈ 3,900토큰 + 다음 판 여유
# 단발은 정책이 고른 상위 5 를 하나도 빼지 않는다 — sources 이벤트의 번호가 먼저 나가서 빠지면 [n] 이 어긋난다.
# 저장 문맥(F4 KorQuAD 794 · 롱테일 11,344 질문)의 문서 칸 어림 최대 3,266 (p99 2,373) + 여유.
# 8192 − grounded 지시(≈3,908) − 질문 칸(≤ 216) ≥ 4,000 이라 단계마다 같은 문서가 들어간다
DOCUMENTS_BUDGET = 4000
QUESTION_BUDGET = 300  # 질문 300자 상한 ≈ 200토큰 + 태그
SUB_QUESTIONS_BUDGET = 300  # 하위 질문 서너 개
QUERIES_BUDGET = 300  # 검색어 최대 3회 × 하위 질문
NOTES_BUDGET = 600  # 반복마다 한 줄씩, 최대 3회
NODE_BUDGET = 800  # 노드별 지시 (agent/decompose@v1 ≈ 600자 · judge@v1 ≈ 900자 ≈ 600토큰)
# 하위 질문마다 근거가 쌓여 단발보다 넓게. 답변 단계에서 8192 − grounded 지시(≈3,908) − 질문 칸 둘(≤ 432) ≥ 3,000 이라
# grounded · partial 이 같은 근거를 받는다 (sources 번호가 단계마다 같아야 한다)
AGENT_DOCUMENTS_BUDGET = 3000
AGENT_LAYOUT_VERSION = "agent_layout@v1"
"""에이전트 답변 프롬프트의 user 턴 배치 판본 — prompt_version 에서 단발의 question@v1 자리. 칸 · 태그를 바꾸면 올린다."""


def ask_layout(budget: int) -> ContextLayout:
    """단발 `/ask` 의 칸: system(지시) → user(문서 · 질문).

    documents 칸 글은 `question@v1` 틀의 `$context` 에 들어간다 (`ask_documents`). 받은 순서(정책이 고른 점수
    내림차순)대로 번호를 붙여 `citations.sources` 와 같은 번호다. question 칸은 틀이 질문을 붙이는 동안 예산을 잡아 두는 자리.
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
                order="given",
            ),
            Slot(QUESTION, "data", budget=QUESTION_BUDGET, tag=QUESTION_TAG, required=True),
        ),
        budget=budget,
        reserved_tags=frozenset({DOCUMENTS_TAG}),
    )


def ask_documents(
    hits: Sequence[Hit], *, system: str = "", question: str = "", budget: int = ASK_BUDGET
) -> str:
    """단발 `/ask` 문서 칸의 글을 돌려준다 — `<document id="n" title="…">` 블록, 본문 속 태그 흉내는 지운다.

    id 는 1부터 hits 순서대로 — 모델은 출처를 이 번호로만 쓰고 서버가 제목 · 링크로 바꾼다 (ADR-026).

    Args:
      hits: 답변 근거로 넘길 검색 결과 (`Decision.context`). 하나도 빼지 않는다 — 부르는 쪽이 이미 번호를 매겼다.
      system: 같이 보낼 지시 글. 예산을 셀 때만 쓴다 (빈 글이면 문서 칸만 잰다).
      question: 같이 보낼 질문. 예산을 셀 때만 쓴다.
      budget: 전체 토큰 상한 (프로필의 `num_ctx`).

    Raises:
      BudgetError: 지시가 칸 예산을 넘거나, 문서를 하나라도 넣지 못했을 때 (예산 부족 · 같은 청크가 두 번).
    """
    candidates = hit_candidates(hits)
    if question:
        candidates.append(Candidate(QUESTION, question))
    instructions = {SYSTEM: system} if system else {}
    context = ContextAssembler(ask_layout(budget), CharTokenCounter()).assemble(
        instructions, candidates
    )
    documents = context.slot(DOCUMENTS)
    if len(documents.placed) != len(hits):
        reasons = sorted({dropped.reason for dropped in context.dropped})
        raise BudgetError(
            f"{DOCUMENTS}: placed {len(documents.placed)} of {len(hits)} hits ({reasons=})"
        )
    return documents.text


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


def agent_answer_messages(
    hits: Sequence[Hit], *, system: str, question: str, budget: int
) -> tuple[Message, ...]:
    """에이전트 답변 단계의 대화를 돌려준다: system(단계 지시) → user(질문 · 근거 · 질문 상기).

    근거는 받은 순서대로 번호 — 부르는 쪽이 sources 이벤트로 이미 보낸 번호다. 하위 질문 · 검색어 · 메모 칸은
    비워 둔다 (에이전트가 쓴 글은 답변 프롬프트에 넣지 않는다). 노드 지시 칸도 비운다 — 지시는 system 에 있다.

    Args:
      hits: 답변 근거 (`Decision.context`). 하나도 빼지 않는다.
      system: 단계 지시 (`build_system` — 단발과 같은 글).
      question: 질문 원문. 질문 칸과 끝의 질문 상기 칸에 넣는다.
      budget: 전체 토큰 상한 (프로필의 `num_ctx`).

    Raises:
      BudgetError: 지시가 칸 예산을 넘거나, 근거를 하나라도 넣지 못했을 때.
    """
    candidates = [
        Candidate(QUESTION, question),
        *hit_candidates(hits),
        Candidate(REMINDER, question),
    ]
    context = ContextAssembler(agent_layout(budget), CharTokenCounter()).assemble(
        {SYSTEM: system}, candidates
    )
    placed = len(context.slot(DOCUMENTS).placed)
    if placed != len(hits):
        reasons = sorted({dropped.reason for dropped in context.dropped})
        raise BudgetError(f"{DOCUMENTS}: placed {placed} of {len(hits)} hits ({reasons=})")
    return context.messages


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
