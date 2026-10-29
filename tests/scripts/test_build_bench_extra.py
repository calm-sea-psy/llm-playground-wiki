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
    assert len(str(year)) == 4
    assert (
        extra.YEAR_MIN_CHANGE <= abs(year - 1976) <= extra.YEAR_WINDOW
    )  # 1년 차이는 오타로 보인다


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
    assert abs(decade - 60) <= extra.DECADE_WINDOW  # 60년대 → 90년대 처럼 멀리 가지 않는다


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
    text = "1985년 대선에서 낙선했다. 1989년에는 국회 행정실장을 역임하였다. 1991년 3당 합당."

    # 1985년과 1991년 사이에서 2년 이상 옮긴 해 (1990년 · 1988년은 1년 차이)
    assert _number("1989년", text) in {"1986년", "1987년"}


def test_substitute_number_skips_a_year_with_no_room_between_neighbors():
    text = "1987년 대선에서 낙선했다. 1989년에는 국회 행정실장을 역임하였다. 1990년 3당 합당."

    assert _number("1989년", text) is None  # 1987년과 1990년 사이에는 1988년뿐 — 1년 차이


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


def test_substitute_number_moves_a_day_by_at_least_the_minimum():
    new = _number("13일")  # 12월 13일 → 12월 14일 은 오타로 보인다

    assert new is not None
    day = int(new.removesuffix("일"))
    assert abs(day - 13) >= extra.DAY_MIN_CHANGE and 1 <= day <= 28


@pytest.mark.parametrize(
    "value", ["1번", "0개", "1.25배", "5·18 광주 민주화 운동", "뉘르부르크링 2009"]
)
def test_substitute_number_refuses_values_it_cannot_change_sensibly(value: str):
    assert _number(value) is None


@pytest.mark.parametrize(
    ("answer", "kind"),
    [("1996년", "date"), ("19-20세기", "date"), ("32년", "number"), ("57 mm", "number")],
)
def test_number_type_treats_two_digit_years_as_durations(answer: str, kind: str):
    assert extra._number_type(answer) == kind


@pytest.mark.parametrize(
    ("name", "style"),
    [
        ("정한용", "ko"),
        ("다치 사쿠타로", "ja"),
        ("시부야 히사타케", "ja"),
        ("허버트 빅스", "western"),
        ("앨버트 로젤리니", "western"),
        ("방석", None),  # 성 없는 두 음절 이름
        ("마커스", None),  # '제임스 마커스' 의 성 — 첫 글자 '마' 가 한국 성이어도 외국 이름
        ("파울 폰 힌덴부르크", None),  # 세 어절
    ],
)
def test_name_style_tells_korean_japanese_and_western_names(name: str, style: str | None):
    assert extra._name_style(name) == style


def test_fictional_names_belong_to_their_own_style():
    for style, names in extra.FICTIONAL_NAMES.items():
        assert {extra._name_style(name) for name in names} == {style}
        assert len(set(names)) == len(names)


def test_substitute_person_keeps_the_style_and_skips_used_or_present_names():
    korean = extra.FICTIONAL_NAMES["ko"]
    used, present = set(korean[:-2]), korean[-2]
    new = extra.substitute_person("정한용", present, random.Random("seed"), used=used)

    assert new == korean[-1]  # 이미 쓴 이름 · 원래 문맥에 있는 이름을 빼면 하나 남는다
    western = extra.substitute_person("허버트 빅스", "", random.Random("seed"))
    assert (
        western in extra.FICTIONAL_NAMES["western"]
    )  # 미국 주지사 자리에 일본 이름이 들어가지 않는다
    assert extra.substitute_person("방석", "", random.Random("seed")) is None


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
    row = extra._counterfactual_row(source, random.Random("0:counterfactual"), skipped)

    assert row is not None and not skipped
    assert row["id"] == "counterfactual:3-0-0"
    assert row["original_answers"] == ["1976년"]
    assert row["answers"] != ["1976년"]
    assert row["substitution"]["type"] == "date"
    # 바뀐 값이 모든 청크에 반영되고 원래 값은 사라진다
    joined = extra.squash("".join(chunk["text"] for chunk in row["context"]))
    assert extra.squash(row["answers"][0]) in joined
    assert extra.squash("1976년") not in joined


def _skip_reason(source: dict) -> str | None:
    skipped = Counter()
    row = extra._counterfactual_row(source, random.Random("0"), skipped)
    return None if row is not None else next(iter(skipped))


def test_counterfactual_skips_a_person_whose_name_is_in_a_chunk_title():
    # 청크 제목은 프롬프트에 실린다 — 본문만 바꾸면 '하비에르 소토마요르' 제목이 답을 알려 준다
    source = _answerable(
        "4-0-0", "높이뛰기 우승자는?", "소토마요르", "person", title="하비에르 소토마요르"
    )

    assert _skip_reason(source) == "answer_in_title"


def test_counterfactual_skips_a_name_followed_by_its_original_script():
    texts = ["우승자는 소토마요르(Sotomayor)였다."] * 5
    source = _answerable("5-0-0", "우승자는?", "소토마요르", "person", texts=texts)

    assert _skip_reason(source) == "name_with_original_script"


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


def test_counterfactual_skips_a_year_with_a_relative_year_nearby():
    # 답의 해를 옮기면 '1981년 시즌 종료 후 … 이듬해 1986년' 처럼 어긋난다
    texts = [
        "1985년 시즌 종료 후 치료를 받아 회복되었지만, 이듬해 1986년 시즌에 다시 쓰러졌다."
    ] * 5
    source = _answerable("11-0-0", "치료를 받은 해는?", "1985년", "date_number", texts=texts)

    assert _skip_reason(source) == "relative_year_nearby"


@pytest.mark.parametrize(
    ("text", "relative"),
    [
        ("이듬해 1986년", True),
        ("같은 해에는", True),
        ("3년 뒤", True),
        ("11년 동안", True),
        ("12개월 동안", True),
        ("그 해결책", False),
        ("1985년 전반기", False),
    ],
)
def test_relative_year_matches_only_relative_expressions(text: str, relative: bool):
    assert (extra._RELATIVE_YEAR.search(text) is not None) == relative


def test_counterfactual_skips_an_answer_also_in_another_document():
    # 다른 문서(노트8)의 800만 화소까지 바꾸면 그 문서의 사실이 틀어진다
    texts = ["전면 카메라는 800만 화소이다."] * 5
    source = _answerable(
        "12-0-0", "전면 카메라 화소는?", "800만", "date_number", title="갤럭시 S8", texts=texts
    )
    source["context"][3]["title"] = "갤럭시 노트8"

    assert _skip_reason(source) == "answer_in_other_document"


def test_counterfactual_changes_a_year_in_other_documents_too():
    # 연도는 다른 문서에서도 대개 같은 사건(루거우차오 사건 · 중일 전쟁)이라 함께 바꾼다
    texts = ["루거우차오 사건은 1937년 7월에 일어났다."] * 5
    source = _answerable(
        "13-0-0", "사건이 일어난 해는?", "1937년", "date_number", title="쇼와 천황", texts=texts
    )
    source["context"][2]["title"] = "중일 전쟁"
    row = extra._counterfactual_row(source, random.Random("0"), Counter())

    assert row is not None
    assert all("1937년" not in chunk["text"] for chunk in row["context"])
    assert row["answers"][0] in row["context"][2]["text"]


def test_counterfactual_skips_items_excluded_by_review():
    item_id = next(iter(extra.COUNTERFACTUAL_EXCLUDE))
    source = _answerable(item_id, "만난 날짜는?", "1940년 6월 21일", "date_number")

    assert _skip_reason(source) == "excluded_by_review"


@pytest.mark.parametrize(
    "other",
    [
        "빅뱅은 2006년 8월 19일 첫 싱글을 발표하였다.",  # 다른 해의 같은 날짜
        "8월 19일부터 오스트리아도 전쟁에 들어갔다.",  # 해가 없는 다른 사실
    ],
)
def test_counterfactual_skips_a_yearless_date_in_two_sentences(other: str):
    texts = ["두 번째 EP 음반을 8월 19일에 발매한다고 발표했다."] * 3 + [other] * 2
    source = _answerable("14-0-0", "EP 음반 발매일은?", "8월 19일", "date_number", texts=texts)

    assert _skip_reason(source) == "yearless_date_repeated"


def test_counterfactual_keeps_a_yearless_date_always_in_the_same_year():
    texts = ["2004년 12월 13일 사형 선고를 받았다."] * 5
    source = _answerable(
        "15-0-0", "사형 선고를 받은 날짜는?", "12월 13일", "date_number", texts=texts
    )

    assert _skip_reason(source) is None


def test_counterfactual_skips_a_year_at_one_end_of_a_range():
    # '취임 1년차(2008년 2월 ~ 2009년 2월)' 의 앞 해만 옮기면 1년차가 3년이 된다
    texts = ["취임 1년차(2008년 2월 ~ 2009년 2월) 2008년 2월 25일 취임식을 했다."] * 5
    source = _answerable("16-0-0", "취임한 해는?", "2008년", "date_number", texts=texts)

    assert _skip_reason(source) == "year_in_range"


def test_counterfactual_skips_a_value_computed_from_others():
    texts = ["체스는 가로 8개와 세로 8개, 총 64개의 정사각형으로 된 판에서 둔다."] * 5
    source = _answerable("17-0-0", "정사각형은 몇 개인가?", "64개", "date_number", texts=texts)

    assert _skip_reason(source) == "computed_value"


def test_substitute_number_keeps_an_ordinal_in_order_across_the_whole_context():
    # 제123대 다이쇼 · 제125대 아키히토가 다른 청크에 있으면 쇼와의 제124대는 옮길 자리가 없다
    texts = [
        "쇼와 천황은 제124대 천황이다.",
        "다이쇼는 제123대 천황이다.",
        "아키히토는 제125대 천황이다.",
    ]

    assert _number("제124대", *texts) is None
    assert _number("제124대", texts[0]) is not None


def test_counterfactual_uses_the_reviewed_value():
    item_id, value = next(iter(extra.COUNTERFACTUAL_REVIEWED.items()))
    texts = ["노르드어는 6개의 파열음이 있다."] * 5
    source = _answerable(item_id, "파열음은 몇 개인가?", "6개", "date_number", texts=texts)
    row = extra._counterfactual_row(source, random.Random("0"), Counter())

    assert row is not None and row["answers"] == [value]


_FAR = "그 뒤로도 여러 대회에 나갔다. " * 20  # 답 근처(NEIGHBOR_WINDOW)를 넘기는 거리


def test_counterfactual_keeps_year_order_across_the_whole_document():
    # '애틀랜타에서 부진한 뒤 … 1997년 실내 시즌' — 멀리 있어도 같은 문서의 연도와 순서를 지킨다
    texts = [f"1996년 애틀랜타 올림픽에서 부진했다. {_FAR}그 뒤 1997년 실내 시즌을 지나갔다."] * 5
    source = _answerable("24-0-0", "부진한 해는?", "1996년", "date_number", texts=texts)
    row = extra._counterfactual_row(source, random.Random("0"), Counter())

    assert row is not None
    assert int(row["answers"][0].removesuffix("년")) < 1997


def test_counterfactual_drops_a_reviewed_year_that_breaks_document_order():
    item_id = "6490156-9-0"  # 외금강 개방 1998년 → 확정 값 2000년
    texts = [f"1998년 외금강 코스를 시작으로 개방했다. {_FAR}1999년 내금강 코스도 열었다."] * 5
    source = _answerable(
        item_id, "외금강 코스가 개방된 해는?", "1998년", "date_number", texts=texts
    )
    row = extra._counterfactual_row(source, random.Random("0"), Counter())

    assert extra.COUNTERFACTUAL_REVIEWED[item_id] == "2000년"
    assert row is not None and int(row["answers"][0].removesuffix("년")) < 1999


def test_counterfactual_uses_a_reviewed_name_unless_already_used():
    item_id = "6557695-3-2"  # 홍명보 → 확정 값 유경훈
    texts = ["차범근은 수비수 홍명보에게까지 공격 가세를 지시했다."] * 5
    source = _answerable(item_id, "공격 가세를 지시한 선수는?", "홍명보", "person", texts=texts)
    reviewed = extra.COUNTERFACTUAL_REVIEWED[item_id]
    row = extra._counterfactual_row(source, random.Random("0"), Counter())
    other = extra._counterfactual_row(source, random.Random("0"), Counter(), used_names={reviewed})

    assert row is not None and row["answers"] == [reviewed]
    assert other is not None and other["answers"] != [reviewed]


def test_build_counterfactual_keeps_values_when_other_items_drop_out():
    # 값은 문항마다 따로 난수 — 앞 문항이 빠져도 남은 문항의 값이 그대로다
    first = _answerable(
        "22-0-0", "입대한 해는?", "1976년", "date_number", texts=["1976년 입대."] * 5
    )
    second = _answerable(
        "23-0-0", "졸업한 해는?", "1981년", "date_number", texts=["1981년 졸업."] * 5
    )
    both, _ = extra._build_counterfactual([first, second], random.Random("0"))
    alone, _ = extra._build_counterfactual([second], random.Random("0"))

    assert both[1]["answers"] == alone[0]["answers"]


def test_build_counterfactual_uses_each_fictional_name_once():
    people = [
        ("19-0-0", "정한용", "김영삼"),
        ("20-0-0", "정운찬", "이명박"),
        ("21-0-0", "안창호", "흥사단"),
    ]
    answerable = [
        _answerable(
            item_id,
            "의혹을 제기한 인물은?",
            name,
            "person",
            title=title,
            texts=[f"{name} 의원이 의혹을 제기했다."] * 5,
        )
        for item_id, name, title in people
    ]
    rows, _ = extra._build_counterfactual(answerable, random.Random("0"))
    names = [row["substitution"]["to"] for row in rows]

    assert len(rows) == len(people) and len(names) == len(set(names))
    assert set(names) <= set(extra.FICTIONAL_NAMES["ko"])


def _spare(item_id, paragraph, year):
    return {
        **_answerable(
            item_id, "졸업한 해는?", f"{year}년", "date_number", texts=[f"{year}년 졸업."] * 5
        ),
        "paragraph": paragraph,
    }


def test_build_counterfactual_fills_from_spare_after_the_source():
    # 1단계 후보를 먼저 다 보고 모자란 만큼 예비 원본에서 — 1단계 문항의 값은 예비 원본과 상관없다
    answerable = [
        _answerable("24-0-0", "입대한 해는?", "1976년", "date_number", texts=["1976년 입대."] * 5)
    ]
    spare = [_spare("25-0-0", "문서A#0", 1981), _spare("26-0-0", "문서B#0", 1985)]
    alone, _ = extra._build_counterfactual(answerable, random.Random("0"))
    filled, _ = extra._build_counterfactual(answerable, random.Random("0"), spare=spare)

    assert [row["id"] for row in filled] == [
        "counterfactual:24-0-0",
        "counterfactual:25-0-0",
        "counterfactual:26-0-0",
    ]
    assert filled[0] == alone[0]


def test_build_counterfactual_takes_one_spare_item_per_paragraph():
    spare = [_spare("27-0-0", "문서A#0", 1981), _spare("28-0-0", "문서A#0", 1985)]
    rows, skipped = extra._build_counterfactual([], random.Random("0"), spare=spare)

    assert len(rows) == 1 and skipped == {"same_paragraph": 1}


def test_counterfactual_skips_when_a_surname_stays_behind():
    texts = ["킴벌리 와이어트가 그룹을 떠났다. 와이어트는 이렇게 말했다."] * 5
    source = _answerable("8-0-0", "그룹을 떠난 사람은?", "킴벌리 와이어트", "person", texts=texts)

    assert _skip_reason(source) == "name_part_left"


def test_counterfactual_skips_a_person_with_a_gender_or_age_cue():
    # 후보 이름의 성별 · 나이를 모르니 '여공 김경숙(당시 21세)' 자리에 아무 이름이나 넣지 않는다
    texts = ["여공 김경숙(당시 21세)이 시위 도중 추락하여 사망했다."] * 5
    source = _answerable("9-0-0", "추락하여 사망한 인물은?", "김경숙", "person", texts=texts)

    assert _skip_reason(source) == "person_cue_nearby"


def test_counterfactual_drops_a_trailing_particle_from_the_answer():
    # KorQuAD 정답 '1976년에' — 조사를 떼고 바꿔야 모델이 '1972년' 만 답해도 맞는다
    texts = ["그는 1976년에 공군에 입대했다."] * 5
    source = _answerable("10-0-0", "언제 입대했나?", "1976년에", "date_number", texts=texts)
    row = extra._counterfactual_row(source, random.Random("0"), Counter())

    assert row is not None
    assert row["substitution"]["from"] == "1976년"
    assert row["answers"][0].endswith("년") and row["original_answers"] == ["1976년에"]
    assert f"{row['answers'][0]}에 공군에" in row["context"][0]["text"]


@pytest.mark.parametrize(
    ("answer", "base"),
    [
        ("1976년에", "1976년"),
        ("1000만을", "1000만"),
        ("1445유로", "1445유로"),
        ("3개월에", "3개월"),
    ],
)
def test_trailing_particle_is_dropped_but_units_stay(answer: str, base: str):
    assert extra._TRAILING_JOSA.sub(r"\1", answer) == base  # '유로' 의 로는 조사가 아니다


@pytest.mark.parametrize(
    ("question", "answer", "expected"),
    [
        ("히로히토에게 국제법을 가르친 사람은?", "다치 사쿠타로", True),
        ("1976년 사고 당시 차종은?", "페라리", False),  # 사람을 묻지 않는다
        (
            "1976년 니키라우다 선수의 전복사고당시 차종은?",
            "페라리",
            False,
        ),  # 선수가 있어도 차종을 묻는다
        ("차범근이 공격에 가세할 것을 지시한 선수는?", "홍명보", True),
        ("북새통을 이룬 곳은 누구의 궁전인가?", "베르사유 궁전", False),  # 장소
        ("누가 계속 신뢰하였는가?", "누리꾼", False),  # 무리
    ],
)
def test_person_items_need_a_person_question_and_a_name(question: str, answer: str, expected: bool):
    assert extra._is_person_item(_answerable("11-0-0", question, answer, "person")) is expected


def test_replace_in_context_respects_digit_boundaries():
    context = [{"rank": 1, "text": "15명과 5명이 남았다."}]
    out, count = extra.replace_in_context(context, "5명", "4명")

    assert count == 1
    assert out[0]["text"] == "15명과 4명이 남았다."


def test_substitute_number_changes_counts_by_at_least_five_percent():
    new = _number("1445유로")

    assert new is not None
    assert abs(int(new.removesuffix("유로")) - 1445) >= 1445 * extra.COUNT_MIN_CHANGE


def test_substitute_number_keeps_the_month_between_nearby_months():
    text = "2013년 7월 28일, 음반을 8월 19일에 발매한다고 발표했다. 그는 9월 말까지 홍보했다."
    new = _number("8월 19일", text)

    assert new is not None and new.startswith("8월 ") and new != "8월 19일"  # 달은 두고 날만


def test_substitute_number_keeps_the_hour_when_other_times_are_nearby():
    text = "오후 11시 50분 즈음 광장을 점령하였다. 오전 12시 2분에 보도하였다."
    new = _number("오전 12시 2분", text)

    assert new is not None and new.startswith("오전 12시 ") and new != "오전 12시 2분"  # 분만


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
