# API 계약 변경 기록

계약(`contracts/openapi.json` — `src/playground_wiki/api/schemas.py`와 라우터에서 만든다)이 바뀌면 `src/playground_wiki/api/version.py`의 `CONTRACT_VERSION`을 올리고 같은 버전의 항목을 여기에 씁니다. 항목 제목은 `## [x.y.z]` 꼴이어야 합니다 (`scripts/build_contract_manifest.py`가 확인합니다).

0.x에서는 삭제 · 이름 변경(breaking)에 minor를, 호환되는 추가 · 설명 변경에 patch를 올립니다. 응답의 값 목록(`stage` · `code` 등)에 값을 더하면 모르는 값을 거절하는 클라이언트는 깨지므로, 그런 변경은 항목에 소비자 영향을 적습니다.

## [0.1.1]

- `GET /health`의 `HealthResponse`에 선택 필드 `qdrant`(`local` · `cloud` · `null`)를 더했습니다. 검색이 붙은 Qdrant의 위치이고, 목업 서버(`RAG_IMPL=mock`)는 `null`입니다. 필드가 없던 0.1.0 응답은 `null`로 읽으면 됩니다.
- 예시 `health.json`에 `"qdrant": null`을 더했습니다 (목업 서버의 응답과 같습니다).

## [0.1.0]

최초 계약.

- HTTP: `POST /search`(`SearchRequest` → `SearchResponse`), `GET /health`(`HealthResponse`), `POST /ask`(`AskRequest`, `?mode=single|agent` → `text/event-stream`).
- `/ask` 이벤트 data: `stage`(`StageEvent`) · `sources`(`SourcesEvent`) · `delta`(`DeltaEvent`) · `done`(`DoneEvent`, 진단 `Diagnostics` · `AgentDiagnostics`) · `error`(`ErrorEvent`). OpenAPI에는 응답의 `itemSchema`(이벤트 이름 · data의 `contentSchema`)와 components로 싣습니다.
- 예시: `/search` 요청 · 응답, `/health`, `/ask` 이벤트 열 7가지 (관문 · refuse · grounded · grounded→partial · error · 에이전트 partial 구간 · 에이전트 못 찾음).
