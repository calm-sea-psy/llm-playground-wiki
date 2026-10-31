"""scripts/ 시험이 같이 쓰는 것 — 스크립트를 모듈로 읽기."""

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).parents[2] / "scripts"


def load_script(name: str) -> ModuleType:
    """scripts/{name}.py 를 모듈로 읽는다 (scripts/ 는 패키지가 아니라 import 로 못 읽는다)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
