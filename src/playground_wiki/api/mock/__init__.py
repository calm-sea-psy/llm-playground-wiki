"""목업 모드(RAG_IMPL=mock): PG · Qdrant · Ollama · GPU · 임베딩/리랭커 모델 없이 서버를 띄운다 (웹 프론트엔드 개발용).

검색은 contracts/examples/search_response.json 의 청크를 돌려주고, 생성은 playground_llm.FakeLLM 이 단계 · 노드마다
고정 답을 낸다. 질문 관문 · 답변 정책 · 캐스케이드 · 에이전트 그래프 · SSE 이벤트는 실제 코드 그대로라 이벤트 순서가
실제와 같다. 질문에 꼬리표를 붙이면 길을 고를 수 있다 (꼬리표 없으면 grounded 로 찾음):

  #일반      1위 점수가 partial 구간 → partial (mode=agent 면 바로 에이전트로)
  #거절      검색 결과 없음 → refuse 고정 문구
  #못찾음    grounded 가 못 찾음 → partial 로 넘어감 (mode=agent 면 에이전트로)
  #생성오류  생성 실패 → error(generation)
  #검색오류  검색 실패 → error(internal), /search 는 500

질문 관문은 실제 규칙이다 — "안녕하세요" 는 잡담, "오늘 날씨 어때" 는 실시간 고정 답.
"""

from playground_wiki.api.mock.llm import GENERATION_ERROR_TAG, NOT_FOUND_TAG, PacedLLM
from playground_wiki.api.mock.search import PARTIAL_TAG, REFUSE_TAG, SEARCH_ERROR_TAG, MockRetriever
from playground_wiki.api.mock.services import MOCK_SEARCH_EXAMPLE, MockServices

__all__ = [
    "GENERATION_ERROR_TAG",
    "MOCK_SEARCH_EXAMPLE",
    "NOT_FOUND_TAG",
    "PARTIAL_TAG",
    "REFUSE_TAG",
    "SEARCH_ERROR_TAG",
    "MockRetriever",
    "MockServices",
    "PacedLLM",
]
