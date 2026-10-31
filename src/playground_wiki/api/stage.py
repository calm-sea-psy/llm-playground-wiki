"""답변 단계 하나(grounded · partial)를 생성한다: 모델 조각 → 첫 줄 표지 떼기 → 출처 번호 거르기 → 보낼 글.

SSE 를 모른다 — 보낼 글 조각(str)과 마지막 결과(`StageOutcome`)만 흘리고, 이벤트로 바꾸고 다음 단계를 고르는 것은
부르는 쪽(`AnswerService`)이다. 단발 /ask 와 에이전트 경로가 같은 생성 조각을 쓴다.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from playground_llm import LLM, DoneReason, Generation, GenerationError, ModelNotFoundError

from playground_wiki.api.threaded_stream import ThreadedStream
from playground_wiki.generation import (
    ChatPrompt,
    CitationStream,
    MarkerReader,
    MarkerStatus,
    Profile,
)


@dataclass(frozen=True)
class StageOutcome:
    """단계 하나를 생성한 결과. 단계 전환 판단 · done 이벤트에 쓴다."""

    sent: bool
    """보낼 글 조각을 하나라도 냈나 — 냈으면 단계를 바꾸지 않는다."""
    marker: MarkerStatus | None
    """첫 줄 표지. 표지를 읽지 않는 단계이거나 판정하기 전에 끊겼으면 None."""
    done_reason: DoneReason | None
    """생성이 멈춘 이유. 시간 초과 · 실패 · 못 찾음으로 끊었거나 서버가 주지 않았으면 None."""
    truncated: bool
    """생성이 max_tokens 에 닿아 멈췄다 (`is_truncated`)."""
    timed_out: bool
    """단계 시간 제한(profile.timeout_sec)에 걸려 끊었다."""
    error: str | None
    """생성 실패의 원인 (서버 로그에만 남긴다). 실패하지 않았으면 None."""
    cited: tuple[int, ...]
    """보낸 답에 남은 출처 번호 (처음 나온 순서, 중복 없이)."""
    dropped: tuple[int, ...]
    """범위를 벗어나 뗀 출처 번호 (나온 순서 그대로)."""


class _StageText:
    """모델 글 조각 → 보낼 글. 표지를 읽는 단계면 첫 줄 표지를 떼고, 출처 번호를 거르고, 첫 글자 전의 공백 조각은 버린다."""

    def __init__(self, reader: MarkerReader | None, citations: CitationStream) -> None:
        self._reader = reader
        self._citations = citations
        self.sent = False
        self.marker: MarkerStatus | None = None

    @property
    def stopped(self) -> bool:
        """표지가 '못 찾음'이라 더 읽지 않는다."""
        return self.marker == "not_found"

    @property
    def cited(self) -> tuple[int, ...]:
        """지금까지 보낸 글에 남은 출처 번호."""
        return self._citations.cited

    @property
    def dropped(self) -> tuple[int, ...]:
        """지금까지 범위를 벗어나 뗀 출처 번호."""
        return self._citations.dropped

    def feed(self, piece: str) -> list[str]:
        """생성 중에 받은 글 조각 하나. 지금 보낼 글 (없으면 빈 목록)."""
        if self._reader is not None:
            if self._reader.result is None:
                if (marked := self._reader.feed(piece)) is None:
                    return []
                self.marker = marked.status
                if marked.status == "not_found":
                    return []
                piece = marked.rest
            else:
                piece = self._reader.body(piece)
        return self._send(self._citations.feed(piece))

    def finish(self, *, truncated: bool) -> list[str]:
        """생성이 끝까지 왔을 때 — 붙잡아 둔 표지 · 출처 번호 꼬리를 정리해 보낼 글."""
        sent: list[str] = []
        if self._reader is not None and self._reader.result is None:
            marked = self._reader.finish()
            self.marker = marked.status
            # 표지를 정하기 전에 한도에 닿아 끝났으면 "표지 없으면 찾음"을 쓰지 않는다 —
            # 붙잡아 둔 글은 끊긴 표지일 수 있으니 못 찾음처럼 아무것도 보내지 않고 partial 로 (ADR-024)
            if marked.status == "not_found" or (marked.status == "missing" and truncated):
                return []
            sent = self._send(self._citations.feed(marked.rest))
        return sent + self._send(self._citations.finish())

    def flush(self) -> list[str]:
        """생성이 중간에 끊겼을 때 — 이미 답을 보내고 있었다면 붙잡아 둔 꼬리(공백 · 출처 번호)까지 보낼 글."""
        return self._send(self._citations.finish()) if self.sent else []

    def _send(self, text: str) -> list[str]:
        # 첫 글자가 나오기 전의 공백만 남은 조각은 보내지 않는다 — 보내면 "답을 보냈다"가 되어 단계를 못 바꾼다
        if not text or (not self.sent and not text.strip()):
            return []
        self.sent = True
        return [text]


async def generate_stage(
    llm: LLM,
    profile: Profile,
    prompt: ChatPrompt,
    *,
    context_count: int,
    read_marker: bool,
    clock: Callable[[], float] = time.perf_counter,
) -> AsyncIterator[str | StageOutcome]:
    """단계 하나를 생성하며 보낼 글 조각(str)을 흘리고, 마지막에 결과(`StageOutcome`)를 한 번 낸다.

    시간 초과 · 생성 실패는 예외로 올리지 않고 결과에 싣는다. 부르는 쪽이 중간에 그만 읽으면(aclose)
    결과 없이 끝나고, 어느 쪽이든 응답 스트림을 닫는다.

    Args:
      llm: 생성할 모델 클라이언트.
      profile: 단계 생성 설정. timeout_sec 는 조각 사이가 아니라 단계 전체의 벽시계 제한이다.
      prompt: `build_prompt` 로 만든 대화.
      context_count: 문맥으로 넘긴 문서 수 — 출처 번호를 1..이 수로 거른다.
      read_marker: 첫 줄 표지를 읽는 단계(grounded@v2)인가. '못 찾음'이면 아무것도 보내지 않고 생성을 끊는다.
      clock: 시간 제한을 잴 시계 (초).
    """
    text = _StageText(MarkerReader() if read_marker else None, CitationStream(context_count))
    deadline = clock() + profile.timeout_sec
    stream = ThreadedStream(llm, profile.spec, prompt.messages)
    # 끝까지 받은 생성. 시간 초과 · 실패 · 못 찾음으로 끊었으면 None
    ended: Generation | None = None
    timed_out, error = False, None
    try:
        final: Generation | None = None
        async with contextlib.aclosing(_model_items(stream, deadline, clock)) as items:
            async for item in items:
                if isinstance(item, Generation):
                    final = item
                    continue
                for piece in text.feed(item):
                    yield piece
                if text.stopped:
                    break  # finally 가 응답 스트림을 닫는다
        if not text.stopped:
            ended = final
            for piece in text.finish(truncated=ended is not None and is_truncated(ended)):
                yield piece
    except TimeoutError:
        timed_out = True
        for piece in text.flush():
            yield piece
    # 모델이 없으면(Ollama 에 pull 하기 전) 첫 호출에서 난다 — 잡지 않으면 done · error 없이 끊긴다
    except (GenerationError, ModelNotFoundError) as e:
        error = str(e)
        for piece in text.flush():
            yield piece
    finally:
        stream.stop()
    yield StageOutcome(
        sent=text.sent,
        marker=text.marker,
        done_reason=None if ended is None else ended.done_reason,
        truncated=ended is not None and is_truncated(ended),
        timed_out=timed_out,
        error=error,
        cited=text.cited,
        dropped=text.dropped,
    )


def is_truncated(generation: Generation) -> bool:
    """생성이 max_tokens 에 닿아 멈췄나 — 서버가 준 done_reason 이 length 인지.

    서버가 done_reason 을 주지 않았으면(None) 생성 토큰 수 ≥ max_tokens 로 짐작한다. None 을 잘리지 않음으로
    보면 그런 서버에서 잘림을 놓친다. stop 이면 토큰 수가 한도와 같아도 잘리지 않은 것이다.
    """
    if generation.done_reason is not None:
        return generation.done_reason == "length"
    max_tokens = generation.model.params.get("max_tokens")
    return max_tokens is not None and generation.usage.completion_tokens >= int(max_tokens)


async def _model_items(
    stream: ThreadedStream, deadline: float, clock: Callable[[], float]
) -> AsyncIterator[str | Generation]:
    """응답 스트림의 답 글 조각(thinking 조각은 뺀다)과 마지막 Generation.

    Raises:
      TimeoutError: 단계 마감(deadline)을 넘겼을 때.
      GenerationError: 생성이 실패했을 때.
      ModelNotFoundError: 서버에 그 모델이 없을 때.
    """
    while True:
        # 단계 전체의 벽시계 제한 — 남은 시간이 없으면 바로 TimeoutError. yield 는 블록 밖에 둔다
        async with asyncio.timeout(max(0.0, deadline - clock())):
            item = await stream.next()
        if item is None:
            return
        if isinstance(item, Generation):
            yield item
        elif item.kind == "text":  # thinking 조각은 화면에 보내지 않는다
            yield item.text
