"""하위 패키지 사이 의존 방향 검사. 검색 엔진은 playground_retrieval 모듈이고, 이 레포는 위키 제품 쪽만 갖는다.

ingest ← sources      (위키 덤프 정제 · 파싱)
generation            (답변 정책 · 생성 설정 — 검색 결과를 받기만 한다)
gate                  (질문 관문 — 잡담 · 실시간 어휘 규칙, 검색 전에 본다)
evaluation            (평가 문항 나누기 · 표본 — 스크립트가 쓴다)
agent ← generation     (LangGraph 에이전트 — 검색 서비스 · LLM 은 받아 쓴다, api 를 모른다)
전부 ← api             (제품 API)

컨텍스트 엔진(칸 · 예산 · 포장)은 playground_llm.context 를 쓴다 (2026-11-13 LLM 레포로 옮김) — 위키에 사본을 두지 않는다.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = "playground_wiki"
SRC = Path(__file__).parents[1] / "src" / PACKAGE

# 하위 패키지 → 가져와도 되는 다른 하위 패키지 (playground_retrieval 은 api 에서 쓴다)
ALLOWED_INTERNAL = {
    "ingest": set(),
    "sources": {"ingest"},
    "generation": set(),
    "gate": set(),
    "evaluation": set(),
    "agent": {"generation"},
    "api": {"ingest", "sources", "generation", "gate", "agent"},
}
# 어디서도 쓰면 안 되는 외부 패키지. api/** 는 pydantic 때문에 ruff TID251 을 꺼 두어 ruff 로는 못 잡는다
BANNED_EVERYWHERE = frozenset(
    {
        "ollama",
        "openai",
        "playground_rag",
    }  # LLM 은 playground_llm 으로만, playground_rag 는 옛 이름
)
# 하위 패키지별로 더 막는 것: 정제 · 파싱은 웹 · 저장소 · 검색을 모른다
BANNED_EXTERNAL = {
    "ingest": {"fastapi", "pydantic", "psycopg", "qdrant_client", "playground_retrieval"},
    "sources": {"fastapi", "pydantic", "psycopg", "qdrant_client", "playground_retrieval"},
    # 설정 파일은 api · scripts 만 읽는다. 정책은 저장소에 붙지 않고 검색 결과(Hit)만 받는다
    "generation": {"fastapi", "pydantic", "psycopg", "qdrant_client", "yaml"},
    "evaluation": {"fastapi", "pydantic", "psycopg", "qdrant_client", "yaml"},
    # 관문은 질문 문자열만 본다. 형태소 분석은 kiwipiepy 를 바로 쓴다 (검색 패키지를 통째로 불러오지 않게)
    "gate": {"fastapi", "pydantic", "psycopg", "qdrant_client", "yaml", "playground_retrieval"},
    # 에이전트는 웹 · 저장소 · 설정 파일을 모른다. pydantic 은 LLM 출력 스키마(schemas.py)에만 — ruff TID251 이 본다
    "agent": {"fastapi", "psycopg", "qdrant_client", "yaml"},
}
# 옮겨 간 컨텍스트 엔진의 옛 자리 — 다시 생기면 playground_llm.context 와 두 벌이 된다
OLD_ENGINE_DIR = ("generation", "context")


def _imports(path: Path) -> list[tuple[int, str]]:
    """파일의 모든 import (함수 안 포함) 를 (줄, 모듈 이름) 으로."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def _violations(package: str, src: Path = SRC) -> list[tuple[str, int, str]]:
    """규칙을 어긴 import 를 (패키지 안 파일 경로, 줄, 모듈 이름) 으로."""
    allowed = ALLOWED_INTERNAL[package] | {package}
    banned = BANNED_EXTERNAL.get(package, set()) | BANNED_EVERYWHERE
    problems = []
    for path in sorted((src / package).rglob("*.py")):
        for line, module in _imports(path):
            parts = module.split(".")
            internal_breach = parts[0] == PACKAGE and len(parts) > 1 and parts[1] not in allowed
            if internal_breach or parts[0] in banned:
                problems.append((path.relative_to(src).as_posix(), line, module))
    return problems


def test_every_subpackage_has_a_rule():
    subpackages = {path.parent.name for path in SRC.glob("*/__init__.py")}

    assert subpackages == set(ALLOWED_INTERNAL)


@pytest.mark.parametrize("package", sorted(ALLOWED_INTERNAL))
def test_subpackage_imports_follow_layers(package: str):
    assert _violations(package) == []


def test_checker_catches_imports_inside_functions(tmp_path: Path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "bad.py").write_text(
        "from playground_wiki.ingest import WikiCleaner\n"
        "def f():\n"
        "    from playground_wiki.api import app\n"
        "    import psycopg\n",
        encoding="utf-8",
    )

    assert _violations("sources", tmp_path) == [
        ("sources/bad.py", 3, "playground_wiki.api"),
        ("sources/bad.py", 4, "psycopg"),
    ]


def test_checker_bans_llm_clients_even_in_api(tmp_path: Path):
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "bad.py").write_text(
        "import pydantic\nfrom ollama import Client\n", encoding="utf-8"
    )

    assert _violations("api", tmp_path) == [("api/bad.py", 2, "ollama")]


def test_context_engine_is_not_copied_back():
    # 폴더가 아니라 .py 를 본다 — 엔진을 옮긴 뒤에도 git 이 지우지 않는 __pycache__ 폴더는 남을 수 있다
    assert not list(SRC.joinpath(*OLD_ENGINE_DIR).rglob("*.py"))
