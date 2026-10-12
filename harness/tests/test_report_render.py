import shutil

import pytest
from typer.testing import CliRunner

from irbench.cli import app
from irbench.report.data import load_results
from irbench.report.render import (
    MARKER,
    build_site,
    engine_slots,
    fmt_bytes,
    fmt_delta,
    fmt_group,
    fmt_ms,
    fmt_score,
)
from report_fixtures import write_bundle

PAGE = "ecommerce__product-search__aws-c7i-4xlarge.html"


def site(tmp_path):
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    out = tmp_path / "site"
    return results, out


def read(out, name):
    return (out / name).read_text(encoding="utf-8")


def test_formatters():
    assert fmt_score(0.48072) == "0.481" and fmt_score(None) == "—"
    assert fmt_delta(0.0123) == "+0.012" and fmt_delta(-0.1) == "−0.100" and fmt_delta(None) == "—"
    assert fmt_ms(2.5) == "2.50" and fmt_ms(27.04) == "27.0" and fmt_ms(None) == "—"
    assert fmt_bytes(727293952) == "727.3 MB" and fmt_bytes(1566838784) == "1.6 GB"
    assert fmt_bytes(512) == "512 B" and fmt_bytes(None) == "—"
    assert fmt_group("keyword/keyword") == "keyword"
    assert fmt_group("filtered/semantic") == "filtered (semantic)"


def test_site_pages_and_matrix(tmp_path):
    results, out = site(tmp_path)
    write_bundle(
        results, "meilisearch", groups={"keyword/keyword": 0.481, "semantic/semantic": 0.5}
    )
    write_bundle(
        results,
        "qdrant",
        groups={"semantic/semantic": 0.55},
        na={"keyword:keyword": "no BM25 in the v0.1 adapter"},
    )
    result = build_site(results, out)

    names = sorted(p.relative_to(out).as_posix() for p in result.pages)
    assert names == sorted(
        [
            "index.html",
            PAGE,
            "runs/ecommerce__product-search__meilisearch__bge-small-en-v1.5__default__aws-c7i-4xlarge.html",
            "runs/ecommerce__product-search__qdrant__bge-small-en-v1.5__default__aws-c7i-4xlarge.html",
        ]
    )
    assert (out / "style.css").exists() and (out / MARKER).exists()
    assert f'href="{PAGE}"' in read(out, "index.html")

    page = read(out, PAGE)
    assert "0.481" in page
    assert 'title="no BM25 in the v0.1 adapter"' in page and ">N/A<" in page
    assert page.index(">keyword<") < page.index(">semantic<")


def test_violation_and_failure_are_explicit(tmp_path):
    results, out = site(tmp_path)
    write_bundle(
        results,
        "meilisearch",
        groups={"sorted/keyword": 0.3},
        violations={"sorted:keyword": "12 order violations"},
    )
    write_bundle(results, "elasticsearch", status="failed", failed_stage="index")
    build_site(results, out)
    page = read(out, PAGE)
    assert "12 order violations" in page and "⚠" in page
    assert "failed: index: RuntimeError: boom" in page


def test_perturbation_breakdown(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"keyword/keyword": 0.5})
    build_site(results, out)
    page = read(out, PAGE)
    # fixture: original 0.55, typo 0.40 (−0.150), prefix 0.30 (−0.250), synonym 0.45 (−0.100)
    assert "0.550" in page and "−0.150" in page and "−0.250" in page and "−0.100" in page


def test_model_comparison_only_with_two_models(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"semantic/semantic": 0.5})
    build_site(results, out)
    assert "Embedding models" not in read(out, PAGE)
    write_bundle(
        results, "meilisearch", model="text-embedding-3-small", groups={"semantic/semantic": 0.6}
    )
    build_site(results, out)
    page = read(out, PAGE)
    assert "Embedding models" in page and "text-embedding-3-small" in page


def test_run_page_has_config_environment_and_credits(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(
        results,
        "qdrant",
        track="tuned",
        config_diff={"index.hnsw_config.m": {"default": None, "effective": 32}},
        credits=[{"name": "Jane Doe", "url": "https://github.com/jane"}],
        na={"hybrid:hybrid": "needs keyword retrieval"},
    )
    build_site(results, out)
    page = read(out, f"runs/{d.name}.html")
    assert "index.hnsw_config.m" in page and "32" in page
    assert '<a href="https://github.com/jane">Jane Doe</a>' in page
    assert "qdrant@sha256:00" in page and "abc123" in page and "20261009" in page
    assert "needs keyword retrieval" in page


def test_local_profile_is_labelled_not_comparable(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", hardware="local")
    build_site(results, out)
    assert "not comparable" in read(out, "ecommerce__product-search__local.html")


def test_text_from_results_is_escaped(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(
        results,
        "meilisearch",
        na={"hybrid:hybrid": "<script>alert(1)</script>"},
        config_diff={"x": {"default": None, "effective": "<b>x</b>"}},
    )
    build_site(results, out)
    for page in (read(out, PAGE), read(out, f"runs/{d.name}.html")):
        assert "<script>alert" not in page and "<b>x</b>" not in page
    assert "&lt;script&gt;" in read(out, PAGE)


def test_empty_results(tmp_path):
    results, out = site(tmp_path)
    build_site(results, out)
    assert "No published results yet" in read(out, "index.html")


def test_rebuild_removes_stale_pages(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "meilisearch")
    build_site(results, out)
    stale = out / "runs" / f"{d.name}.html"
    assert stale.exists()
    shutil.rmtree(d)
    build_site(results, out)
    assert not stale.exists()


def test_refuses_to_overwrite_a_foreign_directory(tmp_path):
    results, out = site(tmp_path)
    out.mkdir()
    (out / "precious.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not written by"):
        build_site(results, out)
    assert (out / "precious.txt").read_text() == "keep me"


def test_engine_slots_are_global_and_stable(tmp_path):
    results, _ = site(tmp_path)
    write_bundle(results, "qdrant", hardware="a")
    write_bundle(results, "meilisearch", hardware="b")
    write_bundle(results, "elasticsearch", hardware="b")
    scenarios, _ = load_results(results)
    assert engine_slots(scenarios) == {"elasticsearch": 1, "meilisearch": 2, "qdrant": 3}
    many = [f"e{i}" for i in range(10)]
    for e in many:
        write_bundle(results, e, hardware="c")
    slots = engine_slots(load_results(results)[0])
    assert max(slots.values()) == 8 and sorted(slots.values()).count(0) == 5


def test_cli_builds_the_site(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    ok = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "3 pages" in ok.output  # index, scenario page, one run page


def test_credit_with_unsafe_scheme_is_not_a_link(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(
        results,
        "qdrant",
        credits=[
            {"name": "Eve", "url": "javascript:alert(1)"},
            {"name": "Jane Doe", "url": "https://github.com/jane"},
        ],
    )
    build_site(results, out)
    page = read(out, f"runs/{d.name}.html")
    assert "javascript:" not in page.lower()
    assert "Eve" in page
    assert '<a href="https://github.com/jane">Jane Doe</a>' in page


def test_out_pointing_at_a_file_fails_cleanly(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    out.write_text("keep me")
    with pytest.raises(FileExistsError, match="not a directory"):
        build_site(results, out)
    assert out.read_text() == "keep me"


def test_cli_exits_1_when_out_is_a_file(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    out.write_text("keep me")
    result = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert result.exit_code == 1
    assert "not a directory" in result.output
    assert out.read_text() == "keep me"


def test_checks_facet_fields_render_dash_for_none(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(
        results,
        "meilisearch",
        checks=[
            {
                "mode": "filtered",
                "base": "keyword",
                "n": 10,
                "filter_violations": 0,
                "order_violations": None,
                "facet_accuracy": None,
                "facet_verified": None,
                "facet_unverifiable": None,
            }
        ],
    )
    build_site(results, out)
    page = read(out, f"runs/{d.name}.html")
    assert ">None<" not in page


def test_cli_exits_1_on_unreadable_bundles(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    bad = write_bundle(results, "qdrant")
    (bad / "scores.json").write_text("{")
    result = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert result.exit_code == 1
    assert bad.name in result.output and (out / "index.html").exists()
    assert bad.name in read(out, "index.html")
