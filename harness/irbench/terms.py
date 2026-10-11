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
