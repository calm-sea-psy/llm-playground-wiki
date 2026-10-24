# llm-playground-wiki

한국어 위키백과 QA 제품입니다. 위키 덤프를 파싱 · 정제해 검색 엔진에 넣고, 위키 검색 서비스(`/search`)와 위키 근거 답변 챗봇(`/ask`)을 제공합니다. 다른 제품(소설 프로젝트 등)은 이 서비스를 HTTP 로 불러 리서치에 씁니다.

공통 기능은 모듈 레포에서 가져다 씁니다.

- [llm-playground-retrieval](https://github.com/calm-sea-psy/llm-playground-retrieval) — 검색 엔진: BM25(Kiwi) · Dense(Qdrant) · 하이브리드 · 리랭커 · 저장소 · 인제스트 흐름 · ANN 지표 (2026-10-23 이 레포에서 분리)
- [llm-playground-corpus](https://github.com/calm-sea-psy/llm-playground-corpus) — 문서 · 청크 타입, 토크나이저, 청커, 임베딩, parquet 입출력
- [llm-playground-llm](https://github.com/calm-sea-psy/llm-playground-llm) — 평가 지표 · KorQuAD 매칭(`playground_evalkit`), LLM 클라이언트 · 프롬프트 라이브러리(`playground_llm`)

## 요구 사항

- [uv](https://docs.astral.sh/uv/) (Python 3.13은 `.python-version`으로 고정)
- Docker (PostgreSQL + Qdrant)
- [Ollama](https://ollama.com/) — `/ask` 의 답 생성. `configs/answer.yaml` 의 모델을 받아 둡니다
- 모듈은 git 태그로 설치됩니다 (retrieval `v0.1.1`, corpus `v0.1.0`, evalkit `evalkit-v0.1.0`. `playground_llm` 은 아직 태그가 없어 커밋으로). 모듈을 함께 고칠 때만 같은 부모 폴더에 나란히 clone 하고 `pyproject.toml` 의 source 를 path 로 바꿉니다

```
작업 폴더/
├── llm-playground-wiki/
├── llm-playground-retrieval/
├── llm-playground-corpus/
└── llm-playground-llm/
```

## 시작하기

```bash
# 의존성 설치
uv sync

# 위키 덤프 다운로드 + 체크섬 확인 (data/dump/)
bash scripts/download_dump.sh

# PostgreSQL · Qdrant 실행 (헬스체크 통과까지 대기)
cp .env.example .env
docker compose up -d --wait

# 테스트 · 린트
uv run pytest
uv run ruff check
```

## 검색 API

인제스트한 컬렉션을 Hybrid(Dense + BM25) → 리랭커 순서로 검색합니다. 시작할 때 임베딩 · 리랭커 모델을 올리고, `data/index/bm25/{컬렉션}` 에 저장한 BM25 색인을 읽습니다 (없으면 PG 청크로 새로 만듭니다 — 서브셋 규모만).

```bash
# BM25 색인을 한 번 만들어 저장 (전체 규모는 수십 분)
uv run python scripts/build_bm25.py --collection kowiki-subset-10000_kure-v1
# 컬렉션 기본값은 configs/ingest.yaml 의 store.collection ({source}_{model}). 서브셋은 직접 지정
RAG_COLLECTION=kowiki-subset-10000_kure-v1 uv run uvicorn playground_wiki.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- `POST /search` — `{"query": "조선의 수도는?", "k": 5, "doc_ids": [], "categories": []}` → 청크 · 리랭커 점수(0~1)
- `GET /health` — 검색 대상 컬렉션 · 색인 청크 수
- Swagger: http://127.0.0.1:8000/docs
- Qdrant 는 gRPC(6334)로 접속합니다. REST 로 바꾸려면 `QDRANT_GRPC=0`
- Qdrant Cloud: `.env.example` 의 Qdrant 항목을 `.env.cloud` 에 채우고(git 제외) `--env-file` 로 넘깁니다. PG 는 로컬 그대로입니다

```bash
uv run --env-file .env.cloud python scripts/ingest.py --parsed data/parsed/kowiki-subset-10000
RAG_COLLECTION=kowiki-subset-10000_kure-v1 uv run --env-file .env.cloud uvicorn playground_wiki.api.app:create_app --factory --port 8000
```

## 답변 API

질문 하나에 위키 근거 답을 SSE(`text/event-stream`)로 흘립니다. 검색 API 와 같은 서버(위 uvicorn 명령)에 붙어 있고, 답은 Ollama 가 만듭니다.

```bash
curl -N http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "경복궁을 중건한 사람은?"}'
```

- `POST /ask` — `{"question": "경복궁을 중건한 사람은?"}`. 앞뒤 공백을 뗀 뒤 1~300자 (`/search` 의 `query` 와 같은 제한). 어기면 스트림 없이 422 JSON 입니다
- 응답은 `text/event-stream` 입니다. 이벤트마다 `event: 이름` 줄과 `data: JSON 한 줄` 뒤에 빈 줄이 옵니다. POST 라 브라우저 `EventSource` 로는 받지 못하니 `fetch` 응답 스트림을 읽어 나눕니다
- 설정은 `configs/answer.yaml` — 질문 게이트 켜기(`gate.enabled`) · 단계 임계값(`thresholds`) · 단계별 모델 · `max_tokens` · 시간 제한(`timeout_sec`). 설정 폴더는 `RAG_CONFIG_DIR`(기본 `configs/`)로 바꿉니다

### 흐름

1. **질문 게이트**(`QueryGate`) — 잡담 · 실시간 질문(날씨 · 시세 · 시각 …)을 형태소 어휘 규칙으로 골라, 검색 없이 고정 답을 보냅니다
2. **검색** — `/search` 와 같은 Hybrid → 리랭커
3. **답변 정책**(`AnswerPolicy`) — 리랭커 1위 점수로 단계를 고릅니다. `thresholds.high` 이상이면 grounded(문서로만 답), `low` 이상이면 partial(문서 내용과 일반 지식을 나눠 답), 그 아래거나 결과가 없으면 refuse(LLM 을 부르지 않고 고정 문구). 근거로 넘기는 문서는 `low` 이상인 것만입니다
4. **grounded → partial** — grounded 는 답 첫 줄에 "찾음" · "못 찾음" 표지를 쓰고, 서버가 읽어 떼어 냅니다. 답 조각을 하나도 보내기 전에 "못 찾음" · 빈 답 · 시간 초과가 나오면 생성을 끊고 partial 로 다시 답합니다 (표지가 없으면 찾음으로 보고 그대로 흘립니다. 표지를 정하기 전에 `max_tokens` 에 닿아 끝났으면 못 찾음처럼 partial 로 넘깁니다)

### 이벤트

| 이벤트 | data | 뜻 |
|---|---|---|
| `stage` | `stage` · `message` | 단계에 들어섰다. `stage` 는 `gate` · `grounded` · `partial` · `refuse`, `message` 는 화면에 띄울 안내 문구 |
| `sources` | `sources`: [`number` · `title` · `section` · `url` · `score`] | 근거로 넘긴 문서. grounded · partial 에서만, 첫 `stage` 앞에 한 번. `number` 가 답 속 `[n]`, `score` 는 리랭커 점수(0~1) |
| `delta` | `text` | 답 조각 — 이어 붙이면 답 전체. grounded 표지와 범위 밖 출처 번호는 서버가 떼고 보냅니다. 질문 게이트 · refuse 는 고정 문구 한 조각 |
| `done` | 아래 표 | 끝. 답을 다 보냈다 |
| `error` | `code` · `message` | 끝. 답 조각을 하나도 못 보낸 채 실패했다. `code` 는 `generation`(LLM 호출 실패) · `timeout`(partial 시간 초과), `message` 는 화면에 띄울 문구 |

순서 (`tests/api/test_ask.py`):

```
질문 게이트에 걸림  stage(gate) → delta → done
refuse              stage(refuse) → delta → done
grounded · partial  sources → stage → delta… → done
grounded → partial  sources → stage(grounded) → stage(partial) → delta… → done
실패                sources → stage(한두 개) → error
```

**답 조각(`delta`)을 하나라도 보낸 뒤에는 단계를 바꾸지 않고 `done` 으로 닫습니다.** grounded → partial 전환은 아무것도 보내기 전에만 일어나고, `error` 는 "아무것도 못 보낸 채 실패"만 뜻합니다. 답을 보내던 중 시간 제한에 걸려도 `error` 가 아니라 `done`(`timed_out: true`)입니다. 그래서 클라이언트는 이미 보여 준 답을 지울 일이 없습니다 — `delta` 를 이어 붙이고, `done` 의 `truncated` · `timed_out` 이 참이면 답이 잘렸다고 표시하면 됩니다.

- partial 이 빈 답이면 refuse 와 같은 고정 문구를 `delta` 로 보내고 `done` 으로 닫습니다
- 답을 보내던 중 LLM 호출이 실패해도 `done` 으로 닫습니다. 지금은 이 경우를 알리는 `done` 필드가 없습니다
- `done` · `error` 없이 스트림이 끝나면(검색 실패 같은 서버 예외) 실패로 처리합니다

| `done` 필드 | 뜻 |
|---|---|
| `stage_path` | 거친 단계 (`["grounded", "partial"]` 등) |
| `cited` | 답에 나온 출처 번호 (처음 나온 순서, 중복 없이) |
| `dropped` | 범위를 벗어나 뗀 출처 번호 |
| `truncated` | 생성이 `max_tokens` 에 닿아 멈췄다(Ollama `done_reason` = length, 서버가 주지 않으면 생성 토큰 수로 짐작) — 답이 중간에 끊겼을 수 있다. 고정 문구(거절)를 보낸 경우는 `false` |
| `timed_out` | 답을 보내던 중 단계 시간 제한(`timeout_sec`, 단계마다 따로)에 걸려 끊겼다 |
| `diagnostics` | 로그 한 줄과 같은 항목 — `stage_path` · `gate`(걸린 규칙 · 종류 · 근거 낱말) · `top_score`(리랭커 1위) · `marker`(grounded 표지: `found` · `not_found` · `missing`) · `spec_id` · `prompt_version` · `question_chars` · `question_sha256` · `timings_ms`(`search` · `first_delta` · `total`) |

예 (값은 예시):

```
event: sources
data: {"sources": [{"number": 1, "title": "경복궁", "section": "역사", "url": "https://ko.wikipedia.org/wiki/%EA%B2%BD%EB%B3%B5%EA%B6%81", "score": 0.97}]}

event: stage
data: {"stage": "grounded", "message": "위키백과 문서에서 답을 찾는 중입니다."}

event: delta
data: {"text": "흥선대원군이"}

event: delta
data: {"text": " 중건했습니다 [1]."}

event: done
data: {"stage_path": ["grounded"], "cited": [1], "dropped": [], "truncated": false, "timed_out": false, "diagnostics": {…}}
```

### 로그

요청마다 `playground_wiki.api.answer` 로거에 INFO 한 줄 `ask {JSON}` 을 남깁니다 (`error` 로 끝난 요청도). 항목은 `done` 의 `diagnostics` 와 같습니다.

- 질문 원문은 기본으로 남기지 않습니다 — 길이(`question_chars`)와 SHA-256(`question_sha256`)만. `RAG_LOG_QUESTIONS=1` 일 때만 `question` 필드에 원문을 더합니다
- `error` 를 보낼 때는 원인을 WARNING 한 줄(`ask error {code}: …`)로 더 남깁니다. 클라이언트에는 화면 문구만 갑니다
- uvicorn 기본 로깅 설정은 uvicorn 로거만 켭니다. 이 INFO 줄을 보려면 로깅 설정(`--log-config`)에서 `playground_wiki` 로거를 INFO 로 켭니다

### 밖으로 나가는 데이터

- 답은 Ollama 가 만듭니다 — 기본 `http://localhost:11434`, `OLLAMA_URL`(또는 `OLLAMA_HOST`)로 바꿉니다. 질문과 검색한 문서는 이 Ollama 로만 가고, 외부 LLM API 로는 나가지 않습니다
- 질의 임베딩 · BM25 · 리랭커는 서버 프로세스 안에서 돕니다
- Qdrant Cloud 를 쓰면 질문마다 질의 벡터만 Qdrant 로 나가고, 질문 원문은 나가지 않습니다. 클라우드에 있는 것은 인제스트 때 올린 청크(위 예시는 서브셋 1만)의 벡터와 payload(청크 · 문서 id, 제목, 섹션, 순번, 분류, 덤프 날짜)뿐이고, 본문은 로컬 PG 에 있습니다

## 구조

```
configs/                 인제스트 · 검색 · 답변 정책 · LLM 설정
src/playground_wiki/
  sources/               WikiSource (덤프 스트리밍 파싱)
  ingest/                WikiCleaner (위키 마크업 정제)
  gate/                  QueryGate (질문 게이트 — 잡담 · 실시간 질문을 검색 전에 걸러 고정 답)
  generation/            AnswerPolicy (리랭커 점수로 grounded · partial · refuse, 단계별 생성 설정) · 프롬프트(prompts/) · 대화 만들기 · 출처 번호 → 제목 · 링크
  evaluation/            평가 문항 나누기(KorQuAD 문단 해시) · 답변 정책 임계값 고르기
  api/                   FastAPI — /search · /health · /ask (검색 엔진은 playground_retrieval, /ask 흐름은 answer.py)
scripts/                 덤프 다운로드 · 파싱 · 인제스트 · BM25 색인 · 평가 · ANN 벤치 · 답변 정책 임계값(calibrate_threshold.py)
tests/
data/                    덤프 · 파싱 결과 · 임베딩 (git 제외)
```

## 주요 결과

한국어 위키백과 2026-10-01 덤프: 문서 488,963 · 청크 2,393,485 (KURE-v1 · 384토큰). 질문은 KorQuAD 1.0 dev 5,766문항.

| 검색기 | hit@5 (전체) | hit@5 (서브셋 1만) | 리포트 |
|---|---|---|---|
| Dense | 0.691 | 0.846 | |
| BM25 (Kiwi) | 0.723 | 0.859 | |
| Hybrid (RRF) | 0.767 | 0.882 | |
| **Hybrid + 리랭커** | **0.834** | **0.911** | [전체](reports/2026-10-23-retrieval-full.md) · [서브셋](reports/2026-10-22-retrieval-d12.md) |

- ANN: FAISS (Flat · HNSW · IVF · SQ8 · PQ) vs Qdrant (float · int8 · int8+rescore) recall–지연 — [reports/ann-benchmark.md](reports/ann-benchmark.md)
- 서브셋 수치는 검색기 사이 상대 비교용이다 (방해 문서가 적어 높게 나온다)

## 진행 상황

- [x] Phase 0 — 골격
- [x] Phase 0.5 — 모듈 분리
- [x] Phase 1 — 인제스트 + 검색 + 모델 선정
- [x] Phase 1.5 — 전체 인제스트
- [ ] Phase 2 — 단발 RAG 챗봇
- [ ] Phase 3 — 에이전트
- [ ] Phase 4 — 웹소설 프로젝트 연결 + 데모
