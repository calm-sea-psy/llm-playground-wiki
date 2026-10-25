"""ruff 금지 import 설정 검사. 프로젝트 ruff.toml 의 banned-api 표는 공유 설정(.ruff/)의 같은 표를 덮어쓰므로
공유 설정의 금지 항목을 빠짐없이 다시 적었는지, 그리고 실제로 ruff 가 잡는지 확인한다."""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROJECT = ROOT / "ruff.toml"
SHARED = ROOT / ".ruff" / "fastapi.toml"
LLM_CLIENTS = ("ollama", "openai")  # LLM 은 playground_llm 으로만 부른다


def _banned(path: Path) -> set[str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    return set(config["lint"]["flake8-tidy-imports"]["banned-api"])


def test_project_bans_keep_shared_bans_and_add_llm_clients():
    project = _banned(PROJECT)

    assert _banned(SHARED) <= project  # .ruff/ 를 새로 복사해 항목이 늘면 여기서 걸린다
    assert set(LLM_CLIENTS) <= project


def test_ruff_reports_banned_imports(tmp_path: Path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import ollama\nimport pydantic\nfrom openai import OpenAI\nimport json\n", encoding="utf-8"
    )

    result = subprocess.run(  # noqa: S603  # 고정 인자로 현재 인터프리터의 ruff 실행
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            str(probe),
            "--config",
            str(PROJECT),
            "--select",
            "TID251",
            "--output-format",
            "concise",
        ],
        capture_output=True,
        encoding="utf-8",  # ruff 는 UTF-8 로 쓴다 — 콘솔 코드 페이지(cp949)로 읽으면 깨진다
        check=False,
    )

    flagged = {line.split("`")[1] for line in result.stdout.splitlines() if "TID251" in line}
    assert flagged == {"ollama", "pydantic", "openai"}
