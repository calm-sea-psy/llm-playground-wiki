"""API 계약 버전. contracts/openapi.json 이 바뀌면 올리고 contracts/CHANGELOG.md 에 같은 버전 항목을 쓴다.

0.x 에서는 삭제 · 이름 변경(breaking)에 minor 를, 호환되는 추가 · 설명 변경에 patch 를 올린다. 1.0 이후는 semver 를 따른다.
OpenAPI 문서의 info.version 이 이 값이다.
"""

CONTRACT_VERSION = "0.1.1"
