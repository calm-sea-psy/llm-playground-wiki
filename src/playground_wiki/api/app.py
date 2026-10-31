"""FastAPI 앱 조립: 시작할 때 저장소 · LLM 에 접속하고 검색 · 답변 서비스를 만든다.

실행:
  uv run uvicorn playground_wiki.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from kiwipiepy import Kiwi
from playground_llm import PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import KiwiAnalyzer, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import (
    ApiConfig,
    load_answer_policy,
    load_dump_date,
    load_query_gate,
)
from playground_wiki.api.routes import ask, search
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import PROMPT_DIR, check_profiles

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """모델 · 색인을 올려 app.state 에 두고, 끝날 때 PG · Ollama 접속을 닫는다.

    설정이 틀렸으면(프롬프트가 없거나 자리표시자가 맞지 않으면) 요청을 받기 전에 여기서 실패한다.
    """
    config = ApiConfig.from_env()
    policy = load_answer_policy(config.config_dir)
    library = PromptLibrary.load(PROMPT_DIR)
    check_profiles(policy.profiles.values(), library)
    # Kiwi 하나가 약 500MB — 질문 관문과 BM25 가 같은 인스턴스를 쓴다 (한 인스턴스를 여러 스레드가 써도 같은 결과)
    kiwi = Kiwi()
    gate = load_query_gate(config.config_dir, KiwiTokenizer(kiwi))
    logger.info("loading search service for collection %s", config.search.collection)
    with PgStore.connect(config.pg) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        service = SearchService.build(
            config.search, pg, QdrantStore.from_env(), analyzer=KiwiAnalyzer(kiwi)
        )
        app.state.search_service = service
        app.state.answer_service = AnswerService(
            search=service,
            policy=policy,
            library=library,
            llm=llm,
            gate=gate,
            dump_date=load_dump_date(config.config_dir),
            log_questions=config.log_questions,
        )
        yield


def create_app() -> FastAPI:
    """라우터를 붙인 앱. lifespan 은 서버가 시작할 때 돈다 (TestClient 를 with 없이 쓰면 돌지 않는다)."""
    app = FastAPI(title="llm-playground-wiki", lifespan=lifespan)
    app.include_router(search.router)
    app.include_router(ask.router)
    return app
