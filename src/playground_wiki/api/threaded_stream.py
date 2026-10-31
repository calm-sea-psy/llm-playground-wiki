"""동기 LLM 스트림을 작업 스레드에서 돌려 이벤트 루프가 조각 단위로 기다리게 한다."""

import asyncio
import contextlib
import threading
from collections.abc import Sequence
from typing import cast

from playground_llm import LLM, Generation, Message, ModelSpec, StreamDelta


class ThreadedStream:
    """동기 `llm.stream` 을 작업 스레드에서 돌려 조각을 asyncio 큐로 넘긴다.

    stop() 하면 작업 스레드가 다음 조각을 받는 즉시 응답 스트림을 닫는다 — 큐 소비만 멈추면 스레드와 GPU 가
    계속 생성한다. 스트림 객체는 그 스레드 안에서만 다룬다 (돌고 있는 제너레이터를 다른 스레드에서 닫을 수 없다).
    이벤트 루프 안에서 만든다 (만들 때 작업 스레드가 바로 시작한다).

    예)
        stream = ThreadedStream(llm, profile.spec, prompt.messages)
        try:
            while (item := await stream.next()) is not None:
                ...
        finally:
            stream.stop()
    """

    _END = object()

    def __init__(self, llm: LLM, spec: ModelSpec, messages: Sequence[Message]) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(llm, spec, messages), daemon=True)
        self._thread.start()

    def _run(self, llm: LLM, spec: ModelSpec, messages: Sequence[Message]) -> None:
        stream = llm.stream(spec, messages)
        try:
            for item in stream:
                if self._stop.is_set():
                    break
                self._put(item)
        except Exception as e:  # noqa: BLE001 — 받는 쪽(이벤트 루프)에 그대로 넘긴다
            self._put(e)
        finally:
            stream.close()  # 응답 스트림을 닫아 Ollama 쪽 생성을 끊는다
            self._put(self._END)

    def _put(self, item: object) -> None:
        # 서버가 내려가 이벤트 루프가 닫혔으면 받을 쪽이 없다
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def next(self) -> StreamDelta | Generation | None:
        """다음 조각. 끝났으면 None. 시간 제한은 부르는 쪽이 `asyncio.timeout` 으로 건다.

        Raises:
          GenerationError: 생성이 실패했을 때 (작업 스레드에서 난 것을 그대로).
          ModelNotFoundError: 서버에 그 모델이 없을 때.
        """
        item = await self._queue.get()
        if item is self._END:
            return None
        if isinstance(item, Exception):
            raise item
        return cast("StreamDelta | Generation", item)

    def stop(self) -> None:
        """생성을 끊는다 (작업 스레드가 다음 조각에서 스트림을 닫는다)."""
        self._stop.set()
