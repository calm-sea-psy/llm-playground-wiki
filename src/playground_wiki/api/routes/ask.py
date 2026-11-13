"""/ask 라우터 — 질문 하나에 대한 답을 SSE 로 흘린다 (ADR-027)."""

from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from playground_wiki.api.answer import AnswerMode
from playground_wiki.api.deps import AnswerServiceDep
from playground_wiki.api.openapi import SSE_MEDIA_TYPE, sse_response
from playground_wiki.api.schemas import ASK_EVENTS, AskRequest

router = APIRouter()
# 프록시 · 브라우저가 조각을 모아 두지 않게
SSE_HEADERS: Mapping[str, str] = MappingProxyType(
    {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
)
ASK_RESPONSE = (
    "text/event-stream. 이벤트마다 `event: 이름` 줄과 `data: JSON 한 줄` 뒤에 빈 줄이 온다. "
    "순서 — 관문 · refuse: stage → delta → done. 답변: sources → stage → delta… → done "
    "(grounded 가 아무것도 못 보내면 stage(grounded) → stage(partial)). 답 조각을 하나도 못 보낸 실패: … → error. "
    "done · error 는 마지막에 한 번만 온다. 둘 다 없이 끝나면 연결이 끊긴 것이다."
)


@router.post(
    "/ask",
    response_class=StreamingResponse,
    responses={status.HTTP_200_OK: sse_response(ASK_EVENTS, description=ASK_RESPONSE)},
)
async def ask(
    request: AskRequest, service: AnswerServiceDep, mode: AnswerMode = "single"
) -> StreamingResponse:
    """질문에 답한다. 이벤트는 stage · sources · delta · done · error 순서로 온다 (text/event-stream).

    `?mode=agent` 면 에이전트(질문 나누기 → 검색 → 판단 …)로 답한다 — sources 앞에 agent_* stage 이벤트가 온다.
    """

    async def events() -> AsyncIterator[str]:
        async for event in service.answer(request.question, mode=mode):
            yield event.to_sse()

    return StreamingResponse(events(), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)
