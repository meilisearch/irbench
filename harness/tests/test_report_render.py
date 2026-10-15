import json
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
from report_fixtures import perf as perf_report
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


def test_run_page_says_when_synonyms_are_not_loaded(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "qdrant", synonyms="no synonym file support")
    loaded = write_bundle(results, "meilisearch")
    build_site(results, out)
    assert "Synonyms not loaded: no synonym file support" in read(out, f"runs/{d.name}.html")
    assert "Synonyms not loaded" not in read(out, f"runs/{loaded.name}.html")


def test_perturbation_table_marks_runs_without_synonyms(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "qdrant", synonyms="no synonym file support")
    write_bundle(results, "meilisearch")
    build_site(results, out)
    page = read(out, PAGE)
    assert page.count("&dagger;") == 2  # one marker on the cell, one on the footnote
    assert 'title="synonyms.txt not loaded: no synonym file support"' in page


def test_run_page_of_an_old_manifest_without_synonyms_key(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "qdrant")
    manifest = json.loads((d / "manifest.json").read_text())
    del manifest["config"]["synonyms"]
    (d / "manifest.json").write_text(json.dumps(manifest))
    build_site(results, out)
    assert "Synonyms not loaded" not in read(out, f"runs/{d.name}.html")


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


def test_performance_section(tmp_path):
    results, out = site(tmp_path)
    write_bundle(
        results,
        "meilisearch",
        perf_report=perf_report(
            {"keyword/keyword": (85.0, 4), "semantic/semantic": (30.0, 2)},
            latency_groups=["keyword/keyword", "semantic/semantic", "filtered/keyword"],
        ),
    )
    write_bundle(
        results,
        "qdrant",
        groups={"semantic/semantic": 0.5},
        na={"keyword:keyword": "no BM25"},
        perf_report=perf_report({"semantic/semantic": (100.0, 8)}, skipped=["keyword/keyword"]),
    )
    write_bundle(results, "elasticsearch", perf_report=None)
    build_site(results, out)
    page = read(out, PAGE)

    assert page.count('<svg class="chart"') == 2  # keyword, semantic for one model
    assert "340" in page  # meilisearch keyword max QPS (85 × 4)
    assert "2.50 / 12.2" in page  # p50 / p99 latency cell
    assert "no BM25" in page  # skipped throughput is N/A with reason
    assert "elasticsearch" in page and "Performance was not measured" in page
    assert "727.3 MB" in page and "1.6 GB" in page and "42.5" in page  # indexing cost


def test_perf_sections_per_page_size(tmp_path):
    results, out = site(tmp_path)
    new = write_bundle(
        results,
        "meilisearch",
        perf_reports={
            20: perf_report({"keyword/keyword": (90.0, 4)}),
            100: perf_report({"keyword/keyword": (50.0, 4)}),
        },
    )
    write_bundle(results, "qdrant", perf_report=perf_report({"keyword/keyword": (10.0, 2)}))
    build_site(results, out)
    page = read(out, PAGE)
    i20 = page.index("Page of 20 results")
    i100 = page.index("Page of 100 results")
    assert i20 < i100
    assert "search results page" in page[i20:i100]
    assert "deep retrieval" in page[i100 : i100 + 200]
    assert "not measured at this page size" in page[i20:i100]
    assert "not measured at this page size" not in page[i100:]
    assert "at 20 results: queries per second" in page

    run_page = read(out, f"runs/{new.name}.html")
    assert run_page.index("Page of 20 results") < run_page.index("Page of 100 results")
    assert run_page.count("Latency p50 / p99 (ms)") == 2


def test_perf_limits_are_rendered_from_the_fixture(tmp_path):
    results, out = site(tmp_path)
    pr = perf_report({"keyword/keyword": (85.0, 4)})
    pr["limits"]["p99_ms"] = 50.0
    write_bundle(results, "meilisearch", perf_report=pr)
    build_site(results, out)
    page = read(out, PAGE)
    assert "50 ms" in page


def test_no_valid_step_is_explicit(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", perf_report=perf_report({"keyword/keyword": (50.0, 0)}))
    build_site(results, out)
    page = read(out, PAGE)
    assert "none" in page and "no concurrency step met" in page


def test_engine_color_follows_the_engine_across_pages(tmp_path):
    results, out = site(tmp_path)
    write_bundle(
        results, "meilisearch", hardware="a", perf_report=perf_report({"keyword/keyword": (1.0, 1)})
    )
    write_bundle(
        results, "qdrant", hardware="b", perf_report=perf_report({"keyword/keyword": (1.0, 1)})
    )
    build_site(results, out)
    assert "var(--series-2)" in read(out, "ecommerce__product-search__b.html")  # qdrant is slot 2


def test_run_page_shows_its_perf(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(
        results, "meilisearch", perf_report=perf_report({"keyword/keyword": (85.0, 2)})
    )
    build_site(results, out)
    page = read(out, f"runs/{d.name}.html")
    assert "Performance" in page and "170" in page


def _edit_json(path, fn):
    path.write_text(json.dumps(fn(json.loads(path.read_text()))))


@pytest.mark.parametrize(
    "corrupt_fn",
    [
        lambda d: _edit_json(d / "manifest.json", lambda m: {**m, "stages": ["prepare"]}),
        lambda d: _edit_json(d / "manifest.json", lambda m: {**m, "config": None}),
        lambda d: _edit_json(
            d / "manifest.json",
            lambda m: {**m, "config": {**m["config"], "credits": [{"url": "https://x"}]}},
        ),
        lambda d: _edit_json(
            d / "scores.json",
            lambda s: {
                **s,
                "relevance": [
                    {k: v for k, v in s["relevance"][0].items() if k != "mode"},
                    *s["relevance"][1:],
                ],
            },
        ),
        lambda d: _edit_json(d / "index.json", lambda i: {**i, "seconds": "42.5"}),
        lambda d: _edit_json(
            d / "scores.json",
            lambda s: {
                **s,
                "not_applicable": {**s["not_applicable"], "renamedmode:keyword": "old bundle"},
            },
        ),
    ],
    ids=[
        "manifest_stages_is_a_list",
        "manifest_config_is_null",
        "credit_without_name",
        "relevance_item_missing_mode",
        "index_seconds_is_a_string",
        "not_applicable_key_has_unknown_mode",
    ],
)
def test_malformed_bundle_shapes_do_not_crash_the_build(tmp_path, corrupt_fn):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    bad = write_bundle(results, "qdrant")
    corrupt_fn(bad)
    result = build_site(results, out)
    assert len(result.unreadable) == 1 and result.unreadable[0].path == bad.name
    assert f'href="{PAGE}"' in read(out, "index.html")  # the healthy sibling still renders


def test_malicious_run_id_does_not_escape_out(tmp_path):
    results, out = site(tmp_path)
    d = write_bundle(results, "meilisearch")
    _edit_json(d / "manifest.json", lambda m: {**m, "run_id": "../../escaped"})
    build_site(results, out)
    assert not (out.parent / "escaped.html").exists()
    assert not (tmp_path / "escaped.html").exists()
    assert not list(out.rglob("escaped.html"))


def test_cli_exits_1_on_unreadable_bundles(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch")
    bad = write_bundle(results, "qdrant")
    (bad / "scores.json").write_text("{")
    result = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert result.exit_code == 1
    assert bad.name in result.output and (out / "index.html").exists()
    assert bad.name in read(out, "index.html")


def _drop_synonym_rows(bundle):
    _edit_json(
        bundle / "scores.json",
        lambda d: {
            **d,
            "relevance": [r for r in d["relevance"] if r["perturbation"] != "synonym"],
        },
    )


def test_scenario_without_synonym_perturbation_has_no_synonym_column(tmp_path):
    results, out = site(tmp_path)
    _drop_synonym_rows(write_bundle(results, "meilisearch", groups={"keyword/keyword": 0.5}))
    build_site(results, out)
    page = read(out, PAGE)
    header = next(h for h in page.split("<thead>")[1:] if "<th>typo</th>" in h).split("</thead>")[0]
    assert "<th>typo</th>" in header and "<th>prefix</th>" in header
    assert "<th>synonym</th>" not in page
    assert "or use a synonym" not in page
    run_page = read(out, "runs/" + next(p.name for p in (out / "runs").glob("*meilisearch*.html")))
    assert "<th>synonym</th>" not in run_page and "<th>typo</th>" in run_page


def test_wands_shaped_bundles_keep_the_synonym_column(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"keyword/keyword": 0.5})
    build_site(results, out)
    assert "<th>synonym</th>" in read(out, PAGE)


def _with_group_rows(bundle, ndcgs):
    def add(d):
        base = next(
            r for r in d["relevance"] if (r["mode"], r["perturbation"]) == ("keyword", "all")
        )
        rows = [
            {**base, "group": g, "n": 10, "metrics": {**base["metrics"], "ndcg@10": v}}
            for g, v in ndcgs.items()
        ]
        return {**d, "relevance": d["relevance"] + rows}

    _edit_json(bundle / "scores.json", add)


def test_query_group_table_has_one_column_per_group(tmp_path):
    results, out = site(tmp_path)
    for engine, q, k in (("meilisearch", 0.61, 0.72), ("qdrant", 0.41, 0.52)):
        bundle = write_bundle(results, engine, groups={"keyword/keyword": 0.5})
        _with_group_rows(bundle, {"question": q, "keyword": k})
    build_site(results, out)
    page = read(out, PAGE)
    assert "By query group" in page
    assert "<th>keyword<br><small>n=10</small></th><th>question<br><small>n=10</small></th>" in page
    assert "n=10" in page  # n per group is shown in the table
    assert "n per group is on the run page" not in page
    assert "typo" in page.split("By query group")[1].split("</p>")[0]
    assert "0.610" in page and "0.720" in page and "0.410" in page and "0.520" in page
    # grouped rows must not leak into the overall matrix
    assert "0.990" not in page


def test_wands_shaped_bundles_have_no_query_group_table(tmp_path):
    results, out = site(tmp_path)
    write_bundle(results, "meilisearch", groups={"keyword/keyword": 0.5})
    build_site(results, out)
    assert "By query group" not in read(out, PAGE)
