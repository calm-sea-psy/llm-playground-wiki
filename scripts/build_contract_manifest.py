"""API 계약 manifest 를 만들고 검사한다. contracts/manifest.json 이 결과물이다.

실행
  uv run python scripts/build_contract_manifest.py                              # 예시 검사 → openapi.json 재생성 → manifest 작성
  uv run python scripts/build_contract_manifest.py --check                      # 아무것도 쓰지 않고 검사만 (CI)
  uv run python scripts/build_contract_manifest.py --check --against origin/main  # 기준 브랜치 대비 버전 변화까지

manifest 가 보증하는 것
  1. 계약 해시: openapi.json 에서 info.version 을 뺀 내용의 해시. 스키마 · 경로 · 이벤트 · 설명 글이 바뀌면 달라진다.
  2. 계약이 바뀌면 CONTRACT_VERSION(src/playground_wiki/api/version.py)을 올리고 contracts/CHANGELOG.md 에 같은 버전
     항목(`## [x.y.z]`)을 써야 한다.
  3. 삭제 · 이름 변경(스키마 · 필드 · 값 · 경로 · 파라미터 · 이벤트)은 breaking 이라 0.x 에서는 minor, 1.0 이후는
     major 를 올려야 한다.
  4. openapi.json · 예시 JSON 이 계약과 맞는다 (손으로 고치거나 재생성을 빠뜨리면 실패).
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from playground_wiki.api.contract import check_example, openapi_text
from playground_wiki.api.version import CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
OPENAPI_PATH = CONTRACTS / "openapi.json"
EXAMPLES_DIR = CONTRACTS / "examples"
MANIFEST_PATH = CONTRACTS / "manifest.json"
CHANGELOG_PATH = CONTRACTS / "CHANGELOG.md"
MANIFEST_SCHEMA = 1  # manifest 자체의 형식 번호. 키를 바꾸면 올린다
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
SHOWN_REMOVED = 6  # 버전 오류 한 줄에 늘어놓는 삭제 항목 수 — 넘치면 … 로 줄인다


# ── 해시 ──────────────────────────────────────────────────────────────
def sha(text: str) -> str:
    """글의 SHA-256 (`sha256:` 앞말)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    """줄 끝을 LF 로 맞춘 글 — 체크아웃 환경(CRLF)과 상관없이 같은 해시가 나오게."""
    return text.replace("\r\n", "\n")


def contract_hash(openapi: Mapping[str, Any]) -> str:
    """info.version 을 뺀 OpenAPI 문서의 해시. 키 차례 · 들여쓰기와 상관없다."""
    body = copy.deepcopy(dict(openapi))
    body.get("info", {}).pop("version", None)
    return sha(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def file_hash(path: Path) -> str:
    """파일 글의 해시 (줄 끝을 맞춘 뒤)."""
    return sha(normalized(path.read_text(encoding="utf-8")))


def example_paths() -> list[Path]:
    """예시 JSON 파일 (이름순)."""
    return sorted(EXAMPLES_DIR.glob("*.json"))


def artifact_hashes() -> dict[str, str]:
    """커밋된 산출물(openapi.json · 예시) → 해시. 키는 레포 루트 기준 경로."""
    paths = [OPENAPI_PATH, *example_paths()]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths if path.exists()}


# ── 구성 요소 (manifest diff 를 사람이 읽고, 삭제 · 이름 변경을 찾는 데 쓴다) ─────────
def components(openapi: Mapping[str, Any]) -> dict[str, Any]:
    """OpenAPI 문서의 스키마 필드 · 값 목록 · 경로 · 파라미터 · SSE 이벤트."""
    schemas: Mapping[str, Any] = openapi.get("components", {}).get("schemas", {})
    fields = {name: sorted(schema.get("properties", {})) for name, schema in schemas.items()}
    enums: dict[str, list[Any]] = {}
    for name, schema in schemas.items():
        for field, prop in schema.get("properties", {}).items():
            if values := _enum_values(prop):
                enums[f"{name}.{field}"] = values
    endpoints: list[str] = []
    parameters: dict[str, list[str]] = {}
    events: list[str] = []
    for path, operations in openapi.get("paths", {}).items():
        for method, operation in operations.items():
            endpoint = f"{method.upper()} {path}"
            endpoints.append(endpoint)
            for parameter in operation.get("parameters", []):
                parameters.setdefault(endpoint, []).append(parameter["name"])
                if values := _enum_values(parameter.get("schema", {})):
                    enums[f"{endpoint}?{parameter['name']}"] = values
            events += [f"{endpoint} {name}" for name in _event_names(operation)]
    return {
        "schemas": fields,
        "enums": enums,
        "endpoints": sorted(endpoints),
        "parameters": parameters,
        "events": events,
    }


def _enum_values(schema: Mapping[str, Any]) -> list[Any]:
    """enum · const 값 (anyOf · items 안까지)."""
    values = list(schema.get("enum", []))
    if "const" in schema:
        values.append(schema["const"])
    for option in schema.get("anyOf", []):
        values += _enum_values(option)
    if "items" in schema:
        values += _enum_values(schema["items"])
    return values


def _event_names(operation: Mapping[str, Any]) -> list[str]:
    """응답 text/event-stream 의 itemSchema 가 적은 이벤트 이름."""
    names: list[str] = []
    for response in operation.get("responses", {}).values():
        for media in response.get("content", {}).values():
            items = media.get("itemSchema", {}).get("oneOf", [])
            names += [item["properties"]["event"]["const"] for item in items]
    return names


def removed_items(base: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """기준 구성 요소에는 있는데 지금은 없는 것 (삭제 · 이름 변경)."""
    out: list[str] = []
    for name, fields in base["schemas"].items():
        if name not in current["schemas"]:
            out.append(f"schema {name}")
            continue
        out += [
            f"field {name}.{field}" for field in fields if field not in current["schemas"][name]
        ]
    for key, values in base["enums"].items():
        out += [
            f"value {key}={value!r}"
            for value in values
            if value not in current["enums"].get(key, [])
        ]
    out += [
        f"endpoint {endpoint}"
        for endpoint in base["endpoints"]
        if endpoint not in current["endpoints"]
    ]
    for endpoint, names in base["parameters"].items():
        now = current["parameters"].get(endpoint, [])
        out += [f"parameter {endpoint}?{name}" for name in names if name not in now]
    out += [f"event {event}" for event in base["events"] if event not in current["events"]]
    return out


# ── 버전 ──────────────────────────────────────────────────────────────
def parse_version(version: str) -> tuple[int, int, int]:
    """x.y.z 를 정수 세 개로.

    Raises:
      ValueError: x.y.z 꼴이 아닐 때.
    """
    match = SEMVER.match(version)
    if match is None:
        raise ValueError(f"version must be x.y.z, got {version!r}")
    return int(match[1]), int(match[2]), int(match[3])


def bump_problems(base: str, current: str, *, changed: bool, removed: list[str]) -> list[str]:
    """기준 버전 → 지금 버전이 계약 변화에 맞게 올랐는지. 어긴 곳을 한 줄씩."""
    before, after = parse_version(base), parse_version(current)
    problems: list[str] = []
    if changed and after <= before:
        problems.append(f"계약이 바뀌었는데 버전이 오르지 않았다 (기준 {base} → 지금 {current})")
    if removed and after > before:
        breaking_ok = after[0] > before[0] if before[0] >= 1 else after[:2] > before[:2]
        if not breaking_ok:
            need = "major" if before[0] >= 1 else "minor (0.x 는 breaking 에 minor)"
            shown = ", ".join(removed[:SHOWN_REMOVED])
            shown += " …" if len(removed) > SHOWN_REMOVED else ""
            problems.append(f"삭제 · 이름 변경이 있어 {need} 를 올려야 한다: {shown}")
    return problems


def changelog_has(text: str, version: str) -> bool:
    """CHANGELOG 글에 `## [version]` 항목이 있는가."""
    return re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE) is not None


def _changelog_text() -> str:
    return CHANGELOG_PATH.read_text(encoding="utf-8") if CHANGELOG_PATH.exists() else ""


def _load_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _git_show(ref: str, path: str) -> str | None:
    """git 기준 ref 의 파일 글. 없으면 None."""
    result = subprocess.run(  # noqa: S603  # 인자는 이 스크립트가 만든 git 명령뿐
        ["git", "show", f"{ref}:{path}"],  # noqa: S607  # PATH 의 git
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _example_problems() -> list[str]:
    problems: list[str] = []
    for path in example_paths():
        problems += [f"{path.name}: {problem}" for problem in check_example(path)]
    return problems


def _version_problems() -> list[str]:
    try:
        parse_version(CONTRACT_VERSION)
    except ValueError as e:
        return [str(e)]
    if not changelog_has(_changelog_text(), CONTRACT_VERSION):
        return [f"contracts/CHANGELOG.md 에 '## [{CONTRACT_VERSION}]' 항목이 없다"]
    return []


# ── build ─────────────────────────────────────────────────────────────
def build() -> int:
    """예시를 검사하고 openapi.json · manifest.json 을 다시 쓴다. 실패하면 1."""
    if problems := _version_problems() + _example_problems():
        for problem in problems:
            print("FAIL", problem)
        return 1
    text = openapi_text()
    OPENAPI_PATH.write_text(text, encoding="utf-8", newline="\n")
    openapi = json.loads(text)
    current_hash, current = contract_hash(openapi), components(openapi)
    old = _load_manifest()
    if old is not None:
        changed = old["contract_hash"] != current_hash
        removed = removed_items(old["components"], current) if changed else []
        if problems := bump_problems(
            old["contract_version"], CONTRACT_VERSION, changed=changed, removed=removed
        ):
            for problem in problems:
                print("FAIL", problem)
            print(
                "     → version.py 의 CONTRACT_VERSION 을 올리고 CHANGELOG.md 에 항목을 쓴 뒤 다시 실행한다"
            )
            print("       (openapi.json 은 이미 다시 썼다)")
            return 1
        if not changed and old["contract_version"] != CONTRACT_VERSION:
            print(f"WARN 계약 내용은 같은데 버전만 {old['contract_version']} → {CONTRACT_VERSION}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_hash": current_hash,
        "components": current,
        "artifacts": artifact_hashes(),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"OK   manifest: v{CONTRACT_VERSION}, {current_hash[:19]}…, 산출물 {len(manifest['artifacts'])}개"
    )
    return 0


# ── check ─────────────────────────────────────────────────────────────
def check(against: str | None) -> int:
    """아무것도 쓰지 않고 manifest · 산출물 · 예시 · (기준 대비) 버전 변화를 검사한다. 실패하면 1."""
    errors = _version_problems() + _example_problems()
    warnings: list[str] = []
    text = openapi_text()
    openapi = json.loads(text)
    current_hash = contract_hash(openapi)
    if not OPENAPI_PATH.exists() or normalized(OPENAPI_PATH.read_text(encoding="utf-8")) != text:
        errors.append("contracts/openapi.json 이 앱과 다르다 (다시 만들지 않았거나 손으로 고쳤다)")
    manifest = _load_manifest()
    if manifest is None:
        errors.append("contracts/manifest.json 이 없다 — build_contract_manifest.py 를 실행한다")
    else:
        if manifest["contract_hash"] != current_hash:
            errors.append(
                "계약이 manifest 와 다르다 — 버전을 올리고 build_contract_manifest.py 를 다시 실행한다"
            )
        if manifest["contract_version"] != CONTRACT_VERSION:
            errors.append(
                f"manifest 버전 {manifest['contract_version']} 과 CONTRACT_VERSION {CONTRACT_VERSION} 이 다르다"
            )
        have, want = artifact_hashes(), manifest["artifacts"]
        if have != want:
            names = sorted(key for key in set(have) | set(want) if have.get(key) != want.get(key))
            errors.append(f"산출물이 manifest 와 다르다: {', '.join(names)}")
    if against is not None and manifest is not None:
        base_text = _git_show(against, MANIFEST_PATH.relative_to(ROOT).as_posix())
        if base_text is None:
            warnings.append(
                f"{against} 에 manifest 가 없다 (처음 들이는 것으로 보고 버전 비교를 건너뛴다)"
            )
        else:
            base = json.loads(base_text)
            changed = base["contract_hash"] != current_hash
            removed = removed_items(base["components"], components(openapi)) if changed else []
            errors += [
                f"{against} 대비: {problem}"
                for problem in bump_problems(
                    base["contract_version"], CONTRACT_VERSION, changed=changed, removed=removed
                )
            ]
    for line in warnings:
        print("WARN", line)
    for line in errors:
        print("FAIL", line)
    print(
        f"{'실패' if errors else '통과'} (오류 {len(errors)}, 경고 {len(warnings)}, 계약 v{CONTRACT_VERSION})"
    )
    return 1 if errors else 0


def main() -> int:
    """커맨드라인 진입점."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="아무것도 쓰지 않고 검사만 한다")
    parser.add_argument(
        "--against", metavar="GIT_REF", help="기준 브랜치와 버전 변화를 비교한다 (--check 와 함께)"
    )
    args = parser.parse_args()
    if args.against and not args.check:
        parser.error("--against 는 --check 와 함께 쓴다")
    return check(args.against) if args.check else build()


if __name__ == "__main__":
    sys.exit(main())
