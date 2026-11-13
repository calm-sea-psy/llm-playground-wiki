"""목업 모드(RAG_IMPL=mock)의 검색 · 답변 서비스 조립."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from playground_llm import FakeLLM
from playground_retrieval.search import SearchService

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import AnswerSetup
from playground_wiki.api.mock.llm import DELAY_SEC, PacedLLM, mock_reply
from playground_wiki.api.mock.search import MockRetriever

# 레포 루트에서 실행할 때의 경로 (configs/ 와 같이)
MOCK_SEARCH_EXAMPLE = Path("contracts/examples/search_response.json")


@dataclass(frozen=True)
class MockServices:
    """목업 모드의 검색 · 답변 서비스. 실제 서버와 같은 클래스에 가짜 검색기 · 가짜 LLM 을 끼웠다.

    예)
        services = MockServices.from_config(setup, top_k=5)
        app.state.search_service = services.search
        app.state.answer_service = services.answer
    """

    search: SearchService
    answer: AnswerService

    @classmethod
    def from_config(
        cls,
        setup: AnswerSetup,
        *,
        top_k: int,
        example: Path = MOCK_SEARCH_EXAMPLE,
        delay_sec: float = DELAY_SEC,
    ) -> Self:
        """설정 폴더에서 읽은 답변 구성(정책 · 프롬프트 · 관문)에 예시 검색기 · FakeLLM 을 붙여 만든다.

        Args:
          setup: 실제 서버와 같은 답변 구성. 임계값으로 예시 점수 구간을 맞춘다.
          top_k: 요청에 k 가 없을 때 돌려주는 청크 수 (retrieval.yaml 의 reranker.top_k).
          example: 검색 결과로 쓸 예시 검색 응답 파일.
          delay_sec: 답 조각 사이 간격 (초). 0 이면 한꺼번에.

        Raises:
          ValueError: 예시에 청크가 없거나 1위 점수가 high 미만일 때.
        """
        retriever = MockRetriever.from_example(example, setup.policy.thresholds)
        search = SearchService(
            retriever, collection=retriever.collection, chunk_count=len(retriever.hits), top_k=top_k
        )
        # FakeLLM 은 받은 호출을 calls 에 쌓는다 — 개발용 서버라 그대로 둔다 (요청 천 건에 수 MB)
        llm = PacedLLM(FakeLLM([mock_reply]), delay_sec=delay_sec)
        return cls(search=search, answer=AnswerService.from_config(setup, search=search, llm=llm))
