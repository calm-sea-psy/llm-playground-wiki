# /ask 응답 명세

`POST /ask`가 흘리는 SSE 이벤트의 계약과 로그 · 외부 전송 범위입니다. 클라이언트는 이 문서를 기준으로 스트림을 읽습니다. 실행 방법과 `/search` · `/health`는 [루트 README](../../../README.md#api)에 있습니다.

이벤트 data의 필드와 값 목록은 `schemas.py`의 `StageEvent` · `SourcesEvent` · `DeltaEvent` · `DoneEvent`(`Diagnostics` · `AgentDiagnostics`) · `ErrorEvent`가 정하고, 서버는 이 모델로 data를 만들어 보냅니다. 같은 모델이 `contracts/openapi.json`의 components로, 경우마다의 이벤트 열이 `contracts/examples/ask_*.json`으로 나갑니다 ([API 계약](../../../README.md#api-계약-contracts)).

## 답변 흐름

응답은 `text/event-stream`입니다. 이벤트마다 `event: 이름` 줄과 `data: JSON 한 줄` 뒤에 빈 줄이 옵니다. POST라 브라우저 `EventSource`로는 받지 못하니 `fetch` 응답 스트림을 읽어 나눕니다.

1. **질문 관문**(`QueryGate`) — 잡담 · 실시간 질문(날씨 · 시세 · 시각 …) · 안전 규칙(자해 · 그루밍)을 형태소 어휘 규칙으로 골라, 검색 없이 고정 답을 보냅니다.
2. **검색** — `/search`와 같은 Hybrid → 리랭커입니다.
3. **답변 정책**(`AnswerPolicy`) — 리랭커 1위 점수로 단계를 고릅니다. `thresholds.high` 이상이면 grounded(문서로만 답), `low` 이상이면 partial(문서 내용과 일반 지식을 나눠 답), 그 아래거나 결과가 없으면 refuse(LLM을 부르지 않고 고정 문구)입니다. 근거로 넘기는 문서는 `low` 이상인 것만입니다.
4. **grounded → partial** — grounded는 답 첫 줄에 "찾음" · "못 찾음" 표지를 쓰고, 서버가 읽어 떼어 냅니다. 답 조각을 하나도 보내기 전에 "못 찾음" · 빈 답 · 시간 초과가 나오면 생성을 끊고 partial로 다시 답합니다. 표지가 없으면 찾음으로 보고 그대로 흘리고, 표지를 정하기 전에 `max_tokens`에 닿아 끝났으면 못 찾음처럼 partial로 넘깁니다.

`?mode=agent`는 단발 먼저, 모자랄 때만 에이전트(LangGraph)로 답합니다. 원문 질문으로 단발과 같이 검색해 1위 점수가 `low` 미만이면 단발과 같은 거절, `high` 이상이면 단발 grounded로 답하고 '찾음'이면 끝납니다. grounded가 아무것도 못 보냈거나 점수가 partial 구간이면 에이전트로 넘어갑니다. 질문을 하위 질문(최대 4개)으로 나눠 검색하고, 모은 문서로 답할 수 있는지 판단해 모자라면 새 검색어로 다시 찾습니다 (검색 반복 최대 3, 원문 검색 결과는 근거로 이어 씀). 답변은 단발과 같은 grounded → partial 캐스케이드입니다.

## 이벤트

| 이벤트 | data | 뜻 |
| --- | --- | --- |
| `stage` | `stage` · `message` | 단계에 들어섬. `stage`는 `gate` · `grounded` · `partial` · `refuse`, 에이전트 모드는 `agent_decompose`(질문 나누기) · `agent_search`(검색) · `agent_judge`(판단)가 더 옴. `message`는 화면에 띄울 안내 문구 |
| `sources` | `sources`: [`number` · `title` · `section` · `url` · `score`] | 근거로 넘긴 문서. `number`가 답 속 `[n]`, `score`는 리랭커 점수(0~1). 문서 칸 예산에 다 안 들어가면 점수 낮은 문서는 빼고 보냄. 여러 번 오면 뒤의 것이 앞의 것을 대신함 |
| `delta` | `text` | 답 조각 — 이어 붙이면 답 전체. grounded 표지와 범위 밖 출처 번호는 서버가 떼고 보냄. 질문 관문 · refuse는 고정 문구 한 조각 |
| `done` | 아래 표 | 끝. 답을 다 보냄 |
| `error` | `code` · `message` | 끝. 답 조각을 하나도 못 보낸 채 실패. `code`는 `generation`(LLM 호출 실패) · `timeout`(partial 시간 초과) · `internal`(검색 실패 같은 그 밖의 서버 실패) |

순서 (`tests/api/test_ask.py` · `tests/api/test_ask_agent.py`):

```
질문 관문에 걸림            stage(gate) → delta → done
refuse                      stage(refuse) → delta → done
grounded · partial          sources → stage → delta… → done
grounded → partial          sources → stage(grounded) → stage(partial) → delta… → done
실패                        sources → stage(한두 개) → error
검색 실패                   error
mode=agent, 단발로 끝남     위 단발 순서 그대로 (관문 · refuse · grounded 찾음)
mode=agent, 못 찾음         sources → stage(grounded) → stage(agent_decompose) → stage(agent_search) → stage(agent_judge) → [agent_search → agent_judge …] → sources(다시, 새 번호) → stage(grounded) → … done
mode=agent, partial 구간    stage(agent_decompose) → stage(agent_search) → stage(agent_judge) → … → sources → stage(grounded 또는 partial) → … done
mode=agent, 안전 규칙에 걸림  … → stage(agent_decompose …) → sources(빈 목록) → stage(gate) → delta → done
```

**답 조각(`delta`)을 하나라도 보낸 뒤에는 단계를 바꾸지 않고 `done`으로 닫습니다.** grounded → partial 전환은 아무것도 보내기 전에만 일어나고, `error`는 "아무것도 못 보낸 채 실패"만 뜻합니다. 그래서 클라이언트는 이미 보여 준 답을 지울 일이 없습니다 — `delta`를 이어 붙이고, `done`의 `truncated` · `timed_out` · `failed`가 참이면 답이 잘렸다고 표시하면 됩니다. partial이 빈 답이면 refuse와 같은 고정 문구를 `delta`로 보내고 `done`으로 닫습니다. `done` · `error` 없이 스트림이 끝나면 연결이 끊긴 것이니 실패로 처리합니다.

| `done` 필드 | 내용 |
| --- | --- |
| `stage_path` | 거친 단계 (`["grounded", "partial"]` 등) |
| `cited` · `dropped` | 답에 나온 출처 번호(처음 나온 순서, 중복 없이) · 범위를 벗어나 뗀 출처 번호 |
| `truncated` | 생성이 `max_tokens`에 닿아 멈춤 (Ollama `done_reason` = length). 고정 문구면 `false` |
| `timed_out` | 답을 보내던 중 단계 시간 제한(`timeout_sec`, 단계마다 따로)에 걸려 끊김 |
| `failed` | 답을 보내던 중 생성이 실패해 끊김. 원인은 서버 로그에만 남음 |
| `diagnostics` | 로그 한 줄과 같은 항목 — `stage_path` · `gate`(걸린 규칙 · 종류 · 근거 낱말) · `top_score` · `marker`(`found` · `not_found` · `missing`) · `done_reason`(단계 → `stop` · `length`) · `spec_id` · `prompt_version` · `question_chars` · `question_sha256` · `timings_ms`(`search` · `first_delta` · `total`) |

에이전트 모드(`mode=agent`)의 `diagnostics`에는 `agent` 칸이 붙습니다 (단발로 끝나도). 하위 질문 · 검색어 글은 질문을 옮겨 담아서 넣지 않습니다 (수만).

| `diagnostics.agent` 필드 | 내용 |
| --- | --- |
| `route` | `gate` · `single_refuse` · `single_found` · `single_error` · `escalated_not_found` · `escalated_partial` · `gate_sub`(넘어간 뒤 하위 질문 · 새 검색어가 안전 규칙에 걸림) |
| `steps` · `timings_ms` | 거친 노드 · 노드별 시간 |
| `decompose` · `sub_questions` | 나누기 상태 `ok` · `fallback` · `error` · 하위 질문 수 |
| `searches` · `iterations` · `seed_hits` | 검색 횟수 · 검색 반복 · 이어 받은 원문 검색 청크 수 |
| `evidence` · `answer_documents` · `answer_top_score` | 모은 청크 수 · 답변 근거 수 · 답변 근거 1위 점수 |
| `judgments` · `stop` | 판단 `sufficient` · `insufficient` · `error` · 멈춘 이유 `sufficient` · `max_iterations` · `no_new_queries` · `judge_error` (그래프를 안 돌았으면 `null`) |
| `prompt_versions` | 노드 프롬프트 |

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
data: {"stage_path": ["grounded"], "cited": [1], "dropped": [], "truncated": false, "timed_out": false, "failed": false, "diagnostics": {…}}
```

## 로그

요청마다 `playground_wiki.api.answer` 로거에 INFO 한 줄 `ask {JSON}`을 남깁니다 (`error`로 끝난 요청도). 항목은 `done`의 `diagnostics`와 같습니다.

- 질문 원문은 기본으로 남기지 않습니다 — 길이(`question_chars`)와 해시(`question_sha256`)만 남깁니다. 해시는 서버를 켤 때마다 새로 뽑는 salt를 키로 한 HMAC-SHA256이라, 같은 실행 안에서만 같은 질문을 묶을 수 있고 짧은 질문도 되짚을 수 없습니다. `RAG_LOG_QUESTIONS=1`(`.env.config`)일 때만 `question` 필드에 원문을 더합니다.
- `error`를 보낼 때는 원인을 WARNING 한 줄(`ask error {code}: …`)로 더 남깁니다. 클라이언트에는 화면 문구만 갑니다.
- uvicorn 기본 로깅 설정은 uvicorn 로거만 켭니다. 이 INFO 줄을 보려면 로깅 설정(`--log-config`)에서 `playground_wiki` 로거를 INFO로 켭니다.

## 밖으로 나가는 데이터

- 답은 Ollama가 만듭니다 — 기본 `http://localhost:11434`, `OLLAMA_URL`(또는 `OLLAMA_HOST`)로 바꿉니다. 질문과 검색한 문서는 이 Ollama로만 가고, 외부 LLM API로는 나가지 않습니다.
- 질의 임베딩 · BM25 · 리랭커는 서버 프로세스 안에서 돕니다.
- Qdrant Cloud를 쓰면 질문마다 질의 벡터만 Qdrant로 나가고(`/search`에 `doc_ids` · `categories`를 주면 그 필터 값도 함께), 질문 원문은 나가지 않습니다. 클라우드에 있는 것은 인제스트 때 올린 청크의 벡터와 payload(청크 · 문서 id, 제목, 섹션, 순번, 분류, 덤프 날짜)뿐이고, 본문은 로컬 PG에 있습니다.
