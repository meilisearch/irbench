ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")

# Row offsets for a staggered QWERTY layout: the row above is shifted right,
# the row below is shifted left, relative to the key's index in its row.
_OFFSETS = ((0, (-1, 1)), (-1, (0, 1)), (1, (-1, 0)))


def _build() -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {}
    for r, row in enumerate(ROWS):
        for i, key in enumerate(row):
            found: set[str] = set()
            for dr, offsets in _OFFSETS:
                rr = r + dr
                if 0 <= rr < len(ROWS):
                    for d in offsets:
                        j = i + d
                        if 0 <= j < len(ROWS[rr]):
                            found.add(ROWS[rr][j])
            neighbors[key] = sorted(found)
    return neighbors


QWERTY_NEIGHBORS = _build()
