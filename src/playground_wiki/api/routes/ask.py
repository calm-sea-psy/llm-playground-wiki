"""/ask 라우터 — 질문 하나에 대한 답을 SSE 로 흘린다 (ADR-027)."""

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from playground_wiki.api.answer import AnswerMode
from playground_wiki.api.deps import AnswerServiceDep
from playground_wiki.api.schemas import AskRequest

router = APIRouter()
# 프록시 · 브라우저가 조각을 모아 두지 않게
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/ask")
async def ask(
    request: AskRequest, service: AnswerServiceDep, mode: AnswerMode = "single"
) -> StreamingResponse:
    """질문에 답한다. 이벤트는 stage · sources · delta · done · error 순서로 온다 (text/event-stream).

    `?mode=agent` 면 에이전트(질문 나누기 → 검색 → 판단 …)로 답한다 — sources 앞에 agent_* stage 이벤트가 온다.
    """

    async def events() -> AsyncIterator[str]:
        async for event in service.answer(request.question, mode=mode):
            yield event.to_sse()

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)
