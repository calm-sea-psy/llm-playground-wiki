"""contracts/examples/*.json 이 계약 모델과 /ask 이벤트 순서 규칙을 지키는지 검사한다 (웹 프론트엔드 목업 데이터의 근거).

실행:  uv run python scripts/check_examples.py
어느 예시가 어떤 모델인지는 파일 이름이 정한다 (playground_wiki.api.contract).
"""

import sys
from pathlib import Path

from playground_wiki.api.contract import EXAMPLES_DIR, check_examples

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """커맨드라인 진입점. 하나라도 어기면 1."""
    results = check_examples(ROOT / EXAMPLES_DIR)
    if not results:
        print(f"FAIL 예시가 없다: {ROOT / EXAMPLES_DIR}")
        return 1
    for name, problems in results.items():
        print(f"{'FAIL' if problems else 'OK  '} {name}")
        for problem in problems:
            print(f"     {problem}")
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
