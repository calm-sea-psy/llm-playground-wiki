"""테스트 공용 fixture."""

import shutil
from pathlib import Path

import pytest

from playground_wiki.evaluation.defense_arms import DEFENSE_BASE

CONFIG_DIR = Path(__file__).parents[1] / "configs"
OPERATING_BASE = "base@v6"
"""configs/answer.yaml 의 운영 base."""


@pytest.fixture
def v4_configs(tmp_path: Path) -> Path:
    """운영 설정의 base 만 방어 칸 기준 판(base@v4)으로 되돌린 폴더 — 칸 도구가 원본으로 받는 모양."""
    out = tmp_path / "v4-configs"
    shutil.copytree(CONFIG_DIR, out)
    answer = out / "answer.yaml"
    text = answer.read_text(encoding="utf-8")
    answer.write_text(
        text.replace(f"system: [{OPERATING_BASE},", f"system: [{DEFENSE_BASE},"), encoding="utf-8"
    )
    return out
