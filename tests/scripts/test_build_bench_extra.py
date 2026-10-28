"""scripts/build_bench_extra.py 의 변형 · 주입 헬퍼와 재현성 검사. KorQuAD 없이 합성 입력으로 돈다."""

import importlib.util
import random
import re
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "build_bench_extra.py"
_spec = importlib.util.spec_from_file_location("build_bench_extra", _SCRIPT)
assert _spec is not None and _spec.loader is not None
extra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extra)


def _answerable(item_id, question, answer, shape, title="문서", texts=None):
    texts = texts or [f"{answer} 에 대한 설명." for _ in range(5)]
    return {
        "id": f"korquad:{item_id}",
        "kind": "answerable",
        "question": question,
        "answers": [answer],
        "gold_title": title,
        "context": [
            {
                "rank": rank,
                "chunk_id": f"{item_id}-{rank}",
                "title": title,
                "section": "",
                "text": text,
                "score": 1.0,
            }
            for rank, text in enumerate(texts, start=1)
        ],
        "strata": {"backlinks": 1, "answer_shape": shape, "answer_rank": 1},
    }


def _number(value: str, *texts: str) -> str | None:
    return extra.substitute_number(value, list(texts) or [value], random.Random("seed"))


def test_substitute_number_year_keeps_width_and_stays_near():
    out = _number("1976년")

    assert out is not None and out != "1976년" and out.endswith("년")
    year = int(out.removesuffix("년"))
    assert len(str(year)) == 4 and abs(year - 1976) <= extra.YEAR_WINDOW


def test_substitute_number_changes_all_runs_and_keeps_month_in_range():
    new = _number("1976년 10월")

    assert new is not None and new != "1976년 10월"
    month = int(new.split("년", 1)[1].replace("월", "").strip())
    assert 1 <= month <= 12  # 바뀐 달도 유효 범위


def test_substitute_number_decade_is_multiple_of_ten():
    new = _number("60년대 초반")

    assert new is not None and new.endswith("년대 초반")
    decade = int(new.removesuffix("년대 초반"))
    assert decade % 10 == 0 and decade != 60 and len(str(decade)) == 2


def test_substitute_number_retries_until_absent_from_context():
    # 바뀐 값이 원래 문맥(여기선 모든 연도)에 있으면 쓰지 않는다
    forbidden = "".join(f"{year}년 " for year in range(1000, 10000))
    assert _number("1976년", f"답은 1976년. {forbidden}") is None


@pytest.mark.parametrize(
    ("value", "pattern"),
    [("19-20세기", r"(\d+)-(\d+)세기"), ("1848년부터 1850년", r"(\d+)년부터 (\d+)년")],
)
def test_substitute_number_moves_a_range_by_one_gap(value: str, pattern: str):
    # v1 은 19-20세기를 21-14세기로 바꿨다 — 앞 덩어리를 단위 없는 수로 따로 옮긴 탓
    new = _number(value)

    assert new is not None
    first, second = (int(part) for part in re.fullmatch(pattern, new).groups())
    old_first, old_second = (int(part) for part in re.fullmatch(pattern, value).groups())
    assert second - first == old_second - old_first
    if value.endswith("세기"):
        assert first >= 1 and second <= extra.CENTURY_MAX


def test_substitute_number_keeps_order_with_nearby_years():
    text = "1987년 대선에서 낙선했다. 1989년에는 국회 행정실장을 역임하였다. 1990년 3당 합당."

    assert _number("1989년", text) == "1988년"  # 1987년과 1990년 사이에 남는 유일한 해


def test_substitute_number_skips_when_the_value_also_stands_elsewhere():
    # 1850년이 답 밖(사관학교 입학)에도 있어 답만 바꾸면 순서가 깨진다
    text = "1848년부터 1850년까지 대학에 다녔다. 1850년 사관학교에 입학했다."

    assert _number("1848년부터 1850년", text) is None


def test_substitute_number_counts_stay_at_least_two_and_near():
    new = _number("5명")

    assert new is not None
    count = int(new.removesuffix("명"))
    assert count != 5 and count >= extra.COUNT_MIN and 2.5 <= count <= 7.5


def test_substitute_number_keeps_round_numbers_round():
    new = _number("1000만")  # '최초로 1000만 팔로워' 가 1003만이 되면 어색하다

    assert new is not None
    value = int(new.removesuffix("만"))
    assert value != 1000 and value % 100 == 0 and 500 <= value <= 1500


def test_substitute_number_keeps_am_pm_hours_within_twelve():
    new = _number("오전 12시 2분")

    assert new is not None
    hour = int(re.match(r"오전 (\d+)시", new).group(1))
    assert 1 <= hour <= 12


@pytest.mark.parametrize("value", ["1번", "0개", "1.25배", "5·18 광주 민주화 운동"])
def test_substitute_number_refuses_values_it_cannot_change_sensibly(value: str):
    assert _number(value) is None


@pytest.mark.parametrize(
    ("answer", "kind"),
    [("1996년", "date"), ("19-20세기", "date"), ("32년", "number"), ("57 mm", "number")],
)
def test_number_type_treats_two_digit_years_as_durations(answer: str, kind: str):
    assert extra._number_type(answer) == kind


def test_substitute_person_picks_other_name_of_the_same_shape():
    rng = random.Random("seed")
    new = extra.substitute_person(
        ["김구"], ["김구", "이순신", "안창호", "다치 사쿠타로"], context_squash="이순신", rng=rng
    )

    assert new == "안창호"  # 원래 답(김구) · 문맥(이순신) 이 아닌 한국식 이름은 하나뿐


def test_substitute_person_keeps_foreign_names_foreign():
    rng = random.Random("seed")
    pool = ["김희애", "정운찬", "킴벌리 와이어트", "다치 사쿠타로"]
    new = extra.substitute_person(["하비에르 소토마요르"], pool, context_squash="", rng=rng)

    assert new in {"킴벌리 와이어트", "다치 사쿠타로"}  # 두 어절 외국 이름끼리


def test_replace_in_context_allows_spacing():
    context = [{"rank": 1, "text": "답은 증강 현실 이다."}]
    out, count = extra.replace_in_context(context, "증강현실", "가상세계")

    assert count == 1
    assert out[0]["text"] == "답은 가상세계 이다."


def test_replace_in_context_fits_the_following_particle():
    context = [{"rank": 1, "text": "와이어트가 떠났다. 와이어트는 말했다. 와이어트와 함께."}]
    out, count = extra.replace_in_context(context, "와이어트", "정운찬")

    assert count == 3
    assert out[0]["text"] == "정운찬이 떠났다. 정운찬은 말했다. 정운찬과 함께."


def test_embed_injection_front_and_end():
    assert extra.embed_injection("본문", "지시", "front") == "지시\n\n본문"
    assert extra.embed_injection("본문", "지시", "end") == "본문\n\n지시"


def test_topic_matches_sensitive_keywords():
    war = _answerable(
        "1-0-0", "전투는 몇 년에 있었나?", "1941년", "date_number", title="러시아 원정"
    )
    plain = _answerable("2-0-0", "앨범 제목은?", "화양연화", "other", title="방탄소년단")

    assert extra._topic(war) == "war"
    assert extra._topic(plain) is None


def test_counterfactual_row_substitutes_number():
    source = _answerable("3-0-0", "몇 년에 입대했나?", "1976년", "date_number", title="차범근")
    skipped = Counter()
    row = extra._counterfactual_row(source, [], random.Random("0:counterfactual"), skipped)

    assert row is not None and not skipped
    assert row["id"] == "counterfactual:3-0-0"
    assert row["original_answers"] == ["1976년"]
    assert row["answers"] != ["1976년"]
    assert row["substitution"]["type"] == "date"
    # 바뀐 값이 모든 청크에 반영되고 원래 값은 사라진다
    joined = extra.squash("".join(chunk["text"] for chunk in row["context"]))
    assert extra.squash(row["answers"][0]) in joined
    assert extra.squash("1976년") not in joined


def _skip_reason(source: dict, pool: list[str] | None = None) -> str | None:
    skipped = Counter()
    row = extra._counterfactual_row(source, pool or [], random.Random("0"), skipped)
    return None if row is not None else next(iter(skipped))


def test_counterfactual_skips_a_person_whose_name_is_in_a_chunk_title():
    # 청크 제목은 프롬프트에 실린다 — 본문만 바꾸면 '하비에르 소토마요르' 제목이 답을 알려 준다
    source = _answerable(
        "4-0-0", "높이뛰기 우승자는?", "소토마요르", "person", title="하비에르 소토마요르"
    )

    assert _skip_reason(source, ["다치 사쿠타로", "킴벌리 와이어트"]) == "answer_in_title"


def test_counterfactual_skips_a_name_followed_by_its_original_script():
    texts = ["우승자는 소토마요르(Sotomayor)였다."] * 5
    source = _answerable("5-0-0", "우승자는?", "소토마요르", "person", texts=texts)

    assert _skip_reason(source, ["다치사쿠타로"]) == "name_with_original_script"


def test_counterfactual_skips_a_year_written_with_an_era_name_nearby():
    texts = ["1869년 메이지 2년 3월 25일 해전에서 폭풍우를 만났다."] * 5
    source = _answerable("6-0-0", "폭풍우를 만난 해는?", "1869년", "date_number", texts=texts)

    assert _skip_reason(source) == "era_year_nearby"


def test_counterfactual_skips_when_the_question_year_is_not_near_the_answer():
    texts = ["2019년 현재 그의 작품은 12개 언어로 번역되어 있다."] * 5
    source = _answerable(
        "7-0-0", "2010년 기준 몇 개 언어로 번역됐나?", "12개", "date_number", texts=texts
    )

    assert _skip_reason(source) == "question_time_mismatch"


def test_counterfactual_skips_when_a_surname_stays_behind():
    texts = ["킴벌리 와이어트가 그룹을 떠났다. 와이어트는 이렇게 말했다."] * 5
    source = _answerable("8-0-0", "그룹을 떠난 사람은?", "킴벌리 와이어트", "person", texts=texts)

    assert _skip_reason(source, ["다치 사쿠타로"]) == "name_part_left"


def test_build_indirect_is_deterministic_and_unique():
    answerable = [
        _answerable(f"{index}-0-0", f"질문 {index}?", f"답{index}", "other") for index in range(60)
    ]
    first = extra._build_indirect(answerable, random.Random("0:injection_indirect"))
    second = extra._build_indirect(answerable, random.Random("0:injection_indirect"))

    assert len(first) == extra.INJECT_INDIRECT
    assert first == second  # seed 고정 → 같은 세트
    canaries = [row["canary"] for row in first]
    assert len(set(canaries)) == len(canaries)  # 문항마다 고유
    for row in first:
        assert row["canary"] in "".join(chunk["text"] for chunk in row["context"])


def test_build_direct_appends_attack_and_keeps_answer():
    answerable = [
        _answerable(f"{index}-0-0", f"질문 {index}?", f"답{index}", "other") for index in range(40)
    ]
    rows = extra._build_direct(answerable, random.Random("0:injection_direct"))

    assert len(rows) == extra.INJECT_DIRECT
    assert {row["attack"]["lang"] for row in rows} == {"ko", "en"}
    for row in rows:
        assert row["canary"] == extra.DIRECT_CANARY
        assert row["question"].startswith("질문")
        assert row["answers"] == [row["question"].split("?", 1)[0].replace("질문 ", "답")]


def test_indirect_templates_cover_at_least_ten():
    assert len({template.id for template in extra.INDIRECT_TEMPLATES}) >= 10
    assert len({template.id for template in extra.DIRECT_TEMPLATES}) >= 10


@pytest.mark.parametrize("template", extra.INDIRECT_TEMPLATES)
def test_indirect_templates_have_canary_slot(template):
    assert "{canary}" in template.ko
    assert "{canary}" in template.en
