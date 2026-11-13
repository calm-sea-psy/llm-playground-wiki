# wiki API 이미지 — /search · /health · /ask 를 서빙한다. 임베딩 · 리랭커는 GPU(CUDA 12.8 빌드 torch)로 돈다.
# 저장소(PostgreSQL · Qdrant) · BM25 색인 · 모델 캐시는 docker-compose.yml 의 api 서비스가 붙이고, 답은 호스트의 Ollama 가 만든다.
#
#   docker compose --profile api up -d --build --wait

FROM python:3.13-slim

# 모듈(playground-*)을 git 태그로 받는다
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 층을 먼저 — 소스만 바뀌면 torch 같은 큰 패키지를 다시 받지 않는다
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY configs ./configs
COPY contracts ./contracts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "playground_wiki.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
