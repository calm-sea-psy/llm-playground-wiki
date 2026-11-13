"""앱의 OpenAPI 문서를 contracts/openapi.json 으로 쓴다. 서버 시작(lifespan) · 저장소 · 모델 없이 돈다.

실행
  uv run python scripts/export_openapi.py                 # contracts/openapi.json
  uv run python scripts/export_openapi.py --out x.json    # 다른 경로로

보통은 build_contract_manifest.py 가 예시 검사 · manifest 작성과 함께 부른다 (README "API 계약").
"""

import argparse
from pathlib import Path

from playground_wiki.api.app import create_app
from playground_wiki.api.contract import OPENAPI_FILE, openapi_text

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """커맨드라인 진입점."""
    parser = argparse.ArgumentParser(description="앱의 OpenAPI 문서를 파일로 쓴다")
    parser.add_argument(
        "--out", type=Path, default=ROOT / OPENAPI_FILE, help=f"쓸 파일 (기본 {OPENAPI_FILE})"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(openapi_text(create_app()), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
