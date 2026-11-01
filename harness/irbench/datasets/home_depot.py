"""`home-depot`: Kaggle "Home Depot Product Search Relevance" products and judgments, with
Google's spelling corrections as the definition of a misspelled query (spec
2026-10-31-irbench-home-depot §4.3-§4.5, §5.1). Internal data: it never leaves the runner's
machine, and nothing from it is committed."""

import ast
import csv
import html
import io
import random
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from irbench.datasets.wands import NormalizedDataset
from irbench.io import write_jsonl, write_qrels
from irbench.models import Qrels

csv.field_size_limit(1 << 30)

# Loader constants: part of the dataset's definition, not a run setting (spec §4.5).
SEED = 20261031
SAMPLE_SIZE = 4000
BRAND_ATTRIBUTE = "MFG Brand Name"

# The Kaggle CSVs are not UTF-8. Latin-1 decodes every byte, so a wrong guess garbles
# accents rather than failing; check the real files when first downloaded.
ENCODING = "latin-1"

MISSPELLED = "misspelled"
CLEAN = "clean"


def grade(average: float) -> int:
    """The raters' average, rounded to the nearest rating and mapped onto 0-2 (spec §4.3)."""
    if average < 1.5:
        return 0
    if average < 2.5:
        return 1
    return 2


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _product_id(value: str) -> str:
    """A product id as the files write it. The attributes file can carry a float-formatted
    id ("100001.0") next to blank rows; both are normalised or skipped by the caller."""
    value = value.strip()
    return value.removesuffix(".0")


def load_spelling(path: Path) -> dict[str, str]:
    """Read the dictionary of whole queries to their corrected queries.

    The file comes from a third party, so it is parsed, never imported or executed. It must be
    exactly one top-level assignment of a dict whose keys and values are string literals;
    anything else raises `ValueError`.
    """
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (SyntaxError, ValueError) as e:  # ValueError: null bytes in the source
        raise ValueError(f"{path}: not a Python file: {e}") from e

    # A leading docstring is an expression statement, not an assignment, and is tolerated.
    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        raise ValueError(f"{path}: expected exactly one top-level assignment")
    assign = body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        raise ValueError(f"{path}: expected an assignment to a single name")
    value = assign.value
    if not isinstance(value, ast.Dict):
        raise ValueError(f"{path}: the assigned value is not a dict literal")
    for key, item in zip(value.keys, value.values, strict=True):
        for node in (key, item):  # a `**` unpacking has a None key
            if not (
                isinstance(node, ast.Constant) and isinstance(node.value, str)
            ):  # a call, a name, a number
                raise ValueError(f"{path}: the dict holds something other than string literals")
    result: dict[str, str] = ast.literal_eval(value)
    return result


def _csv_rows(path: Path) -> Iterator[dict[str, str]]:
    """Rows of the single CSV inside a Kaggle zip, streamed."""
    with zipfile.ZipFile(path) as zf:
        members = [
            n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")
        ]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one CSV, found {members}")
        with zf.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding=ENCODING, newline="")
            yield from csv.DictReader(text)


def _qid_number(qid: str) -> int:
    return int(qid.removeprefix("hd-"))


def normalize_home_depot(raw: dict[str, Path], out_dir: Path) -> NormalizedDataset:
    spelling = load_spelling(raw["spelling"])

    # --- Judgments: one row per (query, product) pair ----------------------------------
    titles: dict[str, tuple[int, str]] = {}  # product -> (smallest row id, title)
    first_row: dict[str, int] = {}  # raw search term -> smallest row id
    pairs: dict[str, dict[str, float]] = {}  # raw search term -> product -> average
    for row in _csv_rows(raw["train"]):
        row_id = int(row["id"])
        pid = _product_id(row["product_uid"])
        term = row["search_term"]
        title = row["product_title"].strip()
        if pid not in titles or row_id < titles[pid][0]:
            titles[pid] = (row_id, title)
        if term not in first_row or row_id < first_row[term]:
            first_row[term] = row_id
        pairs.setdefault(term, {}).setdefault(pid, float(row["relevance"]))

    # --- Docs: every judged product, with brand and description ---------------------------
    judged = set(titles)
    kept = {pid for pid in judged if titles[pid][1]}
    dropped_docs = len(judged) - len(kept)

    brands: dict[str, str] = {}
    for row in _csv_rows(raw["attributes"]):
        if row["name"] != BRAND_ATTRIBUTE:
            continue
        pid = _product_id(row["product_uid"])
        value = row["value"].strip()
        if pid in kept and value and pid not in brands:
            brands[pid] = value

    descriptions: dict[str, str] = {}
    for row in _csv_rows(raw["descriptions"]):
        pid = _product_id(row["product_uid"])
        if pid in kept and pid not in descriptions:
            text = _normalize(html.unescape(row["product_description"]))
            if text:
                descriptions[pid] = text

    def docs() -> Iterator[dict[str, Any]]:
        for pid in sorted(kept, key=int):
            yield {
                "id": pid,
                "title": titles[pid][1],
                "brand": brands.get(pid),
                "description": descriptions.get(pid),
            }

    docs_path = out_dir / "docs.jsonl"
    n_docs = write_jsonl(docs_path, docs())

    # --- Queries: those with at least one relevant product, then the seeded sample --------
    dropped_judgments = 0
    graded: dict[str, dict[str, int]] = {}
    for term, by_product in pairs.items():
        grades: dict[str, int] = {}
        for pid, average in by_product.items():
            if pid not in kept:
                dropped_judgments += 1
                continue
            grades[pid] = grade(average)
        graded[term] = grades

    usable = [term for term, grades in graded.items() if any(g > 0 for g in grades.values())]
    queries_no_positive = len(graded) - len(usable)

    by_qid = {f"hd-{first_row[term]}": term for term in usable}
    eligible = sorted(by_qid, key=_qid_number)
    if len(eligible) > SAMPLE_SIZE:
        sampled = sorted(random.Random(SEED).sample(eligible, SAMPLE_SIZE), key=_qid_number)
    else:
        sampled = eligible

    query_rows: list[dict[str, Any]] = []
    qrels: Qrels = {}
    for qid in sampled:
        term = by_qid[qid]
        text = _normalize(term)
        row: dict[str, Any] = {"qid": qid, "text": text, "group": CLEAN}
        # The dictionary is keyed by the query as typed: match before normalising. A correction
        # that is blank, or equal to the query once whitespace and case are normalised, is no
        # misspelling (it would make an empty or identical variant).
        corrected = _normalize(spelling.get(term, ""))
        if corrected and corrected.lower() != text.lower():
            row["group"] = MISSPELLED
            row["corrected"] = corrected
        query_rows.append(row)
        qrels[qid] = graded[term]

    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    n_queries = write_jsonl(queries_path, query_rows)
    write_qrels(qrels_path, qrels)

    return NormalizedDataset(
        docs_path,
        queries_path,
        qrels_path,
        n_docs,
        n_queries,
        dropped={
            "docs": dropped_docs,
            "judgments": dropped_judgments,
            "queries_no_positive": queries_no_positive,
        },
    )
