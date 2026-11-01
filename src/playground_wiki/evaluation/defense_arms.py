"""간접 인젝션 방어 비교 칸: 운영 base 프롬프트만 바꾼 설정 폴더 사본과, 규칙을 뺀 측정 전용 대조 프롬프트.

칸은 `configs/` 를 칸마다 폴더로 복사해 answer.yaml 의 grounded · partial system 에서 운영 base(SERVICE_BASE)만
바꾼 것이다 — 측정 스크립트는 RAG_CONFIG_DIR 로 그 폴더를 가리킨다 (scripts/eval_defense.py config-dir).
넓힌 후보 `base@v5` 는 운영 프롬프트 모음(generation.PROMPT_DIR)에 있어 채택하면 answer.yaml 한 줄로 고른다.
규칙을 뺀 대조 `base_norule@v4` 는 이 패키지의 CONTROL_PROMPT_DIR 에만 있다 — 서버는 운영 프롬프트 모음만 읽어
(`AnswerSetup.from_config`) answer.yaml 이 가리켜도 시작할 때 멈춘다. 꼬리표를 준 측정 실행만 두 폴더를 합쳐 읽는다
(scripts/eval_injection.py run --tag).
evaluation 은 generation 을 import 하지 않아 프롬프트 이름을 여기 적고, 시험이 프롬프트 파일과 같은지 본다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

ArmName = Literal["base-norule", "base-v5"]
ARM_NAMES: tuple[str, ...] = get_args(ArmName)
CONTROL_PROMPT_DIR = Path(__file__).parent / "control_prompts"
"""측정 전용 프롬프트 (`{이름}/{버전}.md`). 운영 프롬프트 모음 밖이라 서버 · answer.yaml 이 고를 수 없다."""
SERVICE_BASE = "base@v4"
"""지금 운영 base — 칸 사본에서 이 조각만 바꾼다. 방어 전 인젝션 기록(꼬리표 없음)이 이 판으로 쟀다."""
DEFENSE_CONFIG_ROOT = Path("data/eval/defense-configs")
"""칸마다 설정 폴더 사본을 두는 곳 — `{칸 이름}/` (answer.yaml · ingest.yaml · retrieval.yaml)."""
SIDE_ROUND = 1
"""부작용 측정 회차 — 기존 스크립트의 꼬리표는 `{칸 이름}-{회차}` (eval_answer · eval_f4_prose · judge_partial)."""


@dataclass(frozen=True)
class DefenseArm:
    """방어 비교 칸 하나 — 운영 base 대신 넣는 조각."""

    name: ArmName
    """칸 이름. 인젝션 기록 꼬리표(eval_injection --tag)와 설정 폴더 이름으로 그대로 쓴다."""
    base: str
    """운영 base(SERVICE_BASE) 자리에 넣는 프롬프트 조각 `이름@버전`."""
    control: bool
    """측정 전용 대조인지 (CONTROL_PROMPT_DIR 에 있다). False 면 운영 프롬프트 모음의 후보."""
    summary: str
    """리포트에 싣는 한 줄 설명."""

    @property
    def side_tag(self) -> str:
        """부작용 측정 기록의 꼬리표 — `{칸 이름}-{SIDE_ROUND}`."""
        return f"{self.name}-{SIDE_ROUND}"


ARMS: Mapping[str, DefenseArm] = MappingProxyType(
    {
        arm.name: arm
        for arm in (
            DefenseArm(
                "base-norule",
                "base_norule@v4",
                control=True,
                summary=f"`{SERVICE_BASE}` 에서 문서 규칙 줄 하나만 지운 측정 전용 대조",
            ),
            DefenseArm(
                "base-v5",
                "base@v5",
                control=False,
                summary="문서 규칙을 덧붙이기 · 형식 · 링크 · 자칭까지 넓힌 후보",
            ),
        )
    }
)
"""칸 이름 → 칸 (ARM_NAMES 순서 — 시험이 본다)."""


def swap_base(system: Sequence[str], base: str) -> tuple[str, ...]:
    """조각 목록 system 에서 운영 base(SERVICE_BASE) 하나만 base 로 바꾼다.

    Raises:
      ValueError: system 에 SERVICE_BASE 가 정확히 하나가 아닐 때 — 운영 구성이 바뀌었는데 칸이 조용히 다른 것을 바꾸지 않게.
    """
    if list(system).count(SERVICE_BASE) != 1:
        raise ValueError(f"system must hold {SERVICE_BASE} exactly once, got {list(system)}")
    return tuple(base if ref == SERVICE_BASE else ref for ref in system)
