from pathlib import Path


def load_term_groups(path: Path) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        group = [" ".join(t.split()).lower() for t in line.split(",") if t.strip()]
        if len(group) < 2:
            raise ValueError(f"{path}:{lineno}: a group needs at least two terms")
        for term in group:
            if term in seen:
                raise ValueError(
                    f"{path}:{lineno}: term {term!r} already used on line {seen[term]}"
                )
            seen[term] = lineno
        groups.append(group)
    return groups


def scenario_term_groups(path: Path | None) -> list[list[str]]:
    """Term groups of a scenario file, or none when the scenario has no such file."""
    return [] if path is None else load_term_groups(path)


def plain_query_text(text: str) -> str:
    """A query as plain text, identical for every engine.

    `"` (phrase search) and a leading `-` (exclusion) are query operators in Meilisearch,
    Typesense and Postgres `websearch_to_tsquery`, and plain text in Elasticsearch
    `multi_match`, Qdrant and ParadeDB. Removing them keeps one query meaning one thing
    everywhere. Hyphens inside a word (covid-19) are kept.
    """
    words = (w.lstrip("-") for w in text.replace('"', " ").split())
    return " ".join(w for w in words if w)
