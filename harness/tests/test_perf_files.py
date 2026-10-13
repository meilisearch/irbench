import json

from irbench.perf_files import LEGACY_PAGE_SIZE, perf_path, perf_report_paths, read_perf_reports


def test_new_files_are_read_ascending(tmp_path):
    for n in (100, 20):
        perf_path(tmp_path, n).write_text(json.dumps({"n": n}))
    assert list(perf_report_paths(tmp_path)) == [20, 100]
    assert read_perf_reports(tmp_path) == {20: {"n": 20}, 100: {"n": 100}}


def test_legacy_perf_json_is_page_size_100(tmp_path):
    (tmp_path / "perf.json").write_text(json.dumps({"legacy": True}))
    assert read_perf_reports(tmp_path) == {LEGACY_PAGE_SIZE: {"legacy": True}}


def test_new_files_win_over_a_leftover_legacy_file(tmp_path):
    (tmp_path / "perf.json").write_text(json.dumps({"legacy": True}))
    perf_path(tmp_path, 20).write_text(json.dumps({"n": 20}))
    assert read_perf_reports(tmp_path) == {20: {"n": 20}}


def test_no_perf_files(tmp_path):
    assert read_perf_reports(tmp_path) == {}
