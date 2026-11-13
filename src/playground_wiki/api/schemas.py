"""/search · /health · /ask 요청 · 응답 형식과 /ask SSE 이벤트의 data 형식 — 웹 프론트엔드와의 계약이다.

여기와 라우터가 바뀌면 contracts/openapi.json 이 달라진다. 계약 버전을 올리고 scripts/build_contract_manifest.py 로
다시 만든다 (README "API 계약").
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Self

from playground_llm import DoneReason
from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import SearchResult
from playground_retrieval.store.qdrant import QdrantMode
from pydantic import BaseModel, ConfigDict, Field

from playground_wiki.agent import Step, StopReason
from playground_wiki.agent.state import DecomposeStatus, Verdict
from playground_wiki.gate import Kind, Rule
from playground_wiki.generation import MarkerStatus, Stage

MAX_QUERY_CHARS = 300  # 리랭커 입력 = 질의 + 청크(384토큰) ≤ 1024토큰. 한국어 300자는 약 200토큰
MAX_K = 20  # 요청 k 상한 — 리랭커가 매기는 후보 수(retrieval.yaml fusion.top_k)까지만 돌려준다

EventName = Literal["stage", "sources", "delta", "done", "error"]
"""/ask SSE 이벤트 이름."""
AgentStageName = Literal["agent_decompose", "agent_search", "agent_judge"]
"""에이전트 노드의 stage 이름 (agent_ + 노드 이름). 답변 근거 고르기(compose)는 바로 끝나 이벤트가 없다."""
StageName = Literal["gate", Stage, AgentStageName]
"""stage 이벤트의 stage — 질문 관문 · 답변 정책 단계 · 에이전트 노드."""
ErrorCode = Literal["generation", "timeout", "internal"]
"""error 이벤트의 code. internal 은 검색 실패처럼 생성 밖에서 난 실패."""
AgentRoute = Literal[
    "gate",
    "single_refuse",
    "single_found",
    "single_error",
    "escalated_not_found",
    "escalated_partial",
    "gate_sub",
]
"""에이전트 모드가 간 길 (done 진단 agent.route). single_* 는 에이전트 그래프를 돌리지 않고 단발로 끝난 것 —
single_refuse 는 원문 1위 점수가 low 미만, single_found 는 단발 grounded 가 답을 보냄, single_error 는 단발 grounded
생성 실패. escalated_not_found 는 단발 grounded 가 아무것도 못 보내(못 찾음 · 빈 답 · 시간 초과) 에이전트로,
escalated_partial 은 원문 점수가 partial 구간(low ≤ 점수 < high)이라 바로 에이전트로 간 것. gate_sub 는 에이전트로
넘어간 뒤 하위 질문 · 새 검색어가 관문(안전 규칙)에 걸려 그 규칙의 고정 답으로 끝난 것 — 걸린 규칙은 진단의 gate 칸에
원문 관문과 같은 꼴로, 걸린 노드는 agent.steps 의 마지막 노드다."""


class SearchRequest(BaseModel):
    """검색 요청."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(
        min_length=1, max_length=MAX_QUERY_CHARS, description="자연어 질문 또는 검색어"
    )
    k: int | None = Field(
        default=None, ge=1, le=MAX_K, description="돌려받을 청크 수. 없으면 서버 기본값(5)"
    )
    doc_ids: list[str] = Field(default=[], description="이 문서 id 들 안에서만 찾는다")
    categories: list[str] = Field(
        default=[], description="이 분류 중 하나에 속한 문서에서만 찾는다"
    )

    def to_filters(self) -> Filters | None:
        """검색 범위. 제한이 없으면 None."""
        if not self.doc_ids and not self.categories:
            return None
        return Filters(doc_ids=frozenset(self.doc_ids), categories=frozenset(self.categories))


class SearchHit(BaseModel):
    """검색된 청크 한 건."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str = Field(description="원문서 제목")
    section: str = Field(description="섹션 제목. 문서 첫머리면 빈 문자열")
    text: str = Field(description='"제목 > 섹션" 머리줄을 포함한 청크 본문')
    score: float = Field(description="리랭커 관련도 0~1. 질의끼리 비교할 수 있다")

    @classmethod
    def from_hit(cls, hit: Hit) -> Self:
        """검색 결과 한 건을 응답 형식으로."""
        chunk = hit.chunk
        return cls(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            score=hit.score,
        )


class SearchResponse(BaseModel):
    """검색 응답."""

    model_config = ConfigDict(frozen=True)

    query: str
    collection: str
    took_ms: float = Field(description="검색 시간(ms). 앞 요청을 기다린 시간은 빼고")
    hits: list[SearchHit] = Field(description="score 내림차순")

    @classmethod
    def from_result(cls, query: str, collection: str, result: SearchResult) -> Self:
        """서비스 결과를 응답 형식으로."""
        return cls(
            query=query,
            collection=collection,
            took_ms=result.took_ms,
            hits=[SearchHit.from_hit(hit) for hit in result.hits],
        )


class HealthResponse(BaseModel):
    """서버 상태. 웹 프론트엔드 사이드바가 쓴다."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    collection: str
    chunks: int = Field(description="BM25 색인에 올린 청크 수")
    qdrant: QdrantMode | None = Field(
        default=None,
        description="검색이 붙은 Qdrant 의 위치: local(로컬 Docker) · cloud(Qdrant Cloud). 목업 서버는 null",
    )


class AskRequest(BaseModel):
    """답변 요청. 응답은 text/event-stream (stage · sources · delta · done · error)."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, description="자연어 질문")


class StageEvent(BaseModel):
    """stage 이벤트 data — 단계에 들어섬."""

    model_config = ConfigDict(frozen=True)

    stage: StageName = Field(description="들어선 단계. 에이전트 모드는 agent_* 노드가 더 온다")
    message: str = Field(description="화면에 띄울 안내 문구")


class Source(BaseModel):
    """근거로 넘긴 문서 하나."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    number: int = Field(description="답 속 [n] 의 번호. 1부터")
    title: str = Field(description="위키백과 문서 제목")
    section: str = Field(description="섹션 제목. 문서 첫머리면 빈 문자열")
    url: str = Field(description="한국어 위키백과 문서 주소")
    score: float = Field(description="리랭커 관련도 0~1")


class SourcesEvent(BaseModel):
    """sources 이벤트 data — 근거로 넘긴 문서. 여러 번 오면 뒤의 것이 앞의 것을 대신한다."""

    model_config = ConfigDict(frozen=True)

    sources: list[Source] = Field(
        description="number 오름차순. 문서 칸 예산에 다 안 들어가면 점수 낮은 문서는 뺀다. "
        "빈 목록이면 앞서 보낸 출처를 거둔다"
    )


class DeltaEvent(BaseModel):
    """delta 이벤트 data — 답 조각."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="이어 붙이면 답 전체. grounded 표지와 범위 밖 출처 번호는 서버가 떼고 보낸다. "
        "질문 관문 · refuse 는 고정 문구 한 조각"
    )


class GateDiagnostics(BaseModel):
    """질문 관문에 걸린 근거. 질문 원문이 아니라 규칙 사전의 말만 든다."""

    model_config = ConfigDict(frozen=True)

    rule: Rule = Field(description="걸린 규칙")
    kind: Kind = Field(description="규칙 안의 종류")
    matched: list[str] = Field(description="건 근거 낱말")


class Timings(BaseModel):
    """요청 하나에 걸린 시간 (ms). 그 단계를 지나지 않았으면 null."""

    model_config = ConfigDict(frozen=True)

    search: float | None = Field(description="검색. 에이전트 모드는 원문 검색과 검색 노드의 합")
    first_delta: float | None = Field(description="요청부터 첫 답 조각까지")
    total: float | None = Field(description="요청부터 끝까지")


class AgentDiagnostics(BaseModel):
    """에이전트 모드(mode=agent)의 진단. 하위 질문 · 검색어 글은 질문을 옮겨 담아서 넣지 않는다."""

    model_config = ConfigDict(frozen=True)

    route: AgentRoute = Field(description="간 길. single_* 는 그래프를 돌리지 않고 단발로 끝난 것")
    steps: list[Step] = Field(description="거친 노드 (차례대로)")
    decompose: DecomposeStatus | None = Field(
        description="질문 나누기 상태. 그래프를 안 돌았으면 null"
    )
    sub_questions: int = Field(description="하위 질문 수")
    searches: int = Field(description="검색 횟수")
    iterations: int = Field(description="검색 반복 수")
    evidence: int = Field(description="모은 청크 수")
    answer_documents: int = Field(description="답변 근거 수")
    judgments: list[Verdict] = Field(description="판단 노드의 결론 (차례대로)")
    stop: StopReason | None = Field(description="검색 반복을 멈춘 이유. 그래프를 안 돌았으면 null")
    prompt_versions: list[str] = Field(description="노드 프롬프트 (이름@버전:해시)")
    timings_ms: dict[str, float] = Field(description="노드 → 걸린 시간 (ms)")
    seed_hits: int = Field(description="이어 받은 원문 검색 청크 수")
    answer_top_score: float | None = Field(description="답변 근거 1위 점수")


class Diagnostics(BaseModel):
    """요청 하나의 진단 — done 이벤트에 싣고, 같은 항목을 서버 로그 한 줄로 남긴다."""

    model_config = ConfigDict(frozen=True)

    stage_path: list[StageName] = Field(description="거친 단계")
    gate: GateDiagnostics | None = Field(description="질문 관문에 걸렸으면 그 근거")
    top_score: float | None = Field(description="리랭커 1위 점수. 검색하지 않았으면 null")
    marker: MarkerStatus | None = Field(description="grounded 첫 줄 표지. 읽지 않았으면 null")
    done_reason: dict[str, DoneReason | None] = Field(
        description="단계 → 생성이 멈춘 이유. 끊었거나 서버가 주지 않았으면 null"
    )
    spec_id: str | None = Field(description="마지막으로 생성한 단계의 모델 설정")
    prompt_version: str | None = Field(description="마지막으로 생성한 단계의 프롬프트")
    question_chars: int = Field(description="질문 글자 수")
    question_sha256: str = Field(
        description="질문의 HMAC-SHA256 (서버를 켤 때마다 바뀌는 키). 질문 원문은 싣지 않는다"
    )
    timings_ms: Timings
    agent: AgentDiagnostics | None = Field(
        default=None,
        exclude_if=lambda agent: agent is None,
        description="mode=agent 일 때만 있다. 단발이면 키가 없다",
    )


class DoneEvent(BaseModel):
    """done 이벤트 data — 끝. 답을 다 보냈다 (잘렸으면 truncated · timed_out · failed 가 참)."""

    model_config = ConfigDict(frozen=True)

    stage_path: list[StageName] = Field(description="거친 단계")
    cited: list[int] = Field(description="답에 나온 출처 번호 (처음 나온 순서, 중복 없이)")
    dropped: list[int] = Field(description="범위를 벗어나 뗀 출처 번호")
    truncated: bool = Field(description="생성이 max_tokens 에 닿아 멈춤. 고정 문구면 false")
    timed_out: bool = Field(description="답을 보내던 중 단계 시간 제한에 걸려 끊김")
    failed: bool = Field(description="답을 보내던 중 생성이 실패해 끊김. 원인은 서버 로그에만")
    diagnostics: Diagnostics


class ErrorEvent(BaseModel):
    """error 이벤트 data — 끝. 답 조각을 하나도 못 보낸 채 실패했다."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode = Field(
        description="generation: LLM 호출 실패 · timeout: partial 시간 초과 · internal: 그 밖의 서버 실패"
    )
    message: str = Field(description="화면에 띄울 문구. 서버 사정은 싣지 않는다")


AskEventData = StageEvent | SourcesEvent | DeltaEvent | DoneEvent | ErrorEvent
ASK_EVENTS: Mapping[EventName, type[AskEventData]] = MappingProxyType(
    {
        "stage": StageEvent,
        "sources": SourcesEvent,
        "delta": DeltaEvent,
        "done": DoneEvent,
        "error": ErrorEvent,
    }
)
"""/ask 이벤트 이름 → data 모델. OpenAPI 의 /ask 응답과 components 가 이것으로 만들어진다."""
EVENT_NAMES: Mapping[type[AskEventData], EventName] = MappingProxyType(
    {model: name for name, model in ASK_EVENTS.items()}
)
"""data 모델 → 이벤트 이름 (`ASK_EVENTS` 를 뒤집은 것)."""
