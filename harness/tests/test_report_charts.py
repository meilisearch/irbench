import pytest

from irbench.report.charts import Point, Series, nice_max, qps_chart

STEPS = [1, 2, 4, 8]


def series(label, slot, qps, invalid_last=False, dashed=False):
    pts = [
        Point(c, q, not (invalid_last and i == len(qps) - 1), f"{c}: {q}")
        for i, (c, q) in enumerate(zip(STEPS, qps))
    ]
    return Series(label, slot, dashed, pts)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, 1),
        (0.4, 0.5),
        (85.6, 100),
        (170.7, 200),
        (2426, 2500),
        (2950, 5000),
        (5000, 5000),
    ],
)
def test_nice_max(value, expected):
    assert nice_max(value) == pytest.approx(expected)


def test_two_series():
    svg = str(
        qps_chart(
            "keyword",
            [
                series("meilisearch", 2, [85, 170, 300, 310], True),
                series("qdrant", 3, [100, 200, 390, 700]),
            ],
            STEPS,
        )
    )
    assert svg.startswith("<svg") and 'role="img"' in svg and 'aria-label="keyword"' in svg
    assert svg.count("<polyline") == 2 and svg.count("<circle") == 8
    assert svg.count('class="marker invalid"') == 1
    assert "var(--series-2)" in svg and "var(--series-3)" in svg
    assert svg.count('class="direct"') == 2  # ≤ 4 series: direct labels
    for tick in ("1", "2", "4", "8"):
        assert f">{tick}</text>" in svg
    assert ">1,000</text>" in svg  # y axis tops at nice_max(700) = 1000


def test_many_series_drop_direct_labels():
    svg = str(qps_chart("x", [series(f"e{i}", i, [10, 20]) for i in range(1, 6)], STEPS))
    assert 'class="direct"' not in svg


def test_single_point_has_a_marker_and_no_line():
    svg = str(qps_chart("x", [series("solo", 1, [42.0])], STEPS))
    assert "<polyline" not in svg and svg.count("<circle") == 1


def test_folded_and_dashed_series():
    svg = str(qps_chart("x", [series("e9", 0, [1, 2], dashed=True)], STEPS))
    assert "var(--series-other)" in svg and "dashed" in svg


def test_labels_are_escaped():
    svg = str(qps_chart("<t>", [Series("<b>x</b>", 1, False, [Point(1, 1.0, True, "<i>")])], STEPS))
    assert "<b>" not in svg and "<i>" not in svg and "&lt;b&gt;" in svg


def test_nothing_to_plot():
    assert qps_chart("x", [], STEPS) is None
    assert qps_chart("x", [Series("a", 1, False, [])], STEPS) is None
