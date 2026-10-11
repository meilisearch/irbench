from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCT_TSV = (
    "product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description"
    "\tproduct_features\trating_count\taverage_rating\treview_count\n"
    '0\tsolid wood platform bed\tBeds\tFurniture / Bedroom Furniture / Beds\ta "sturdy" bed'
    "\tcolor : brown|material : wood\t15.0\t4.5\t15.0\n"
    "1\tgrey velvet sofa\tSofas|Sectionals\tFurniture / Living Room Furniture / Sofas"
    "\tsoft sofa\tcolor : grey\t\t\t\n"
    "2\toak coffee table\tCoffee & Cocktail Tables\tFurniture / Living Room Furniture / Tables"
    "\tround table\tmaterial : oak\t3.0\t3.0\t3.0\n"
    "3\tmodern nightstand\tNightstands\tFurniture / Bedroom Furniture / Nightstands"
    "\ttwo drawers\t\t8.0\t4.0\t8.0\n"
)
QUERY_TSV = (
    "query_id\tquery\tquery_class\n"
    "0\tplatform bed\tBeds\n"
    "1\tvelvet couch\tSofas\n"
    "2\tsalon chair\tMassage Chairs\n"
)
LABEL_TSV = (
    "id\tquery_id\tproduct_id\tlabel\n"
    "0\t0\t0\tExact\n"
    "1\t0\t3\tPartial\n"
    "2\t1\t1\tExact\n"
    "3\t1\t2\tIrrelevant\n"
    "4\t2\t2\tIrrelevant\n"
)


@pytest.fixture
def wands_raw(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "wands_raw"
    raw.mkdir()
    files = {"product": PRODUCT_TSV, "query": QUERY_TSV, "label": LABEL_TSV}
    out = {}
    for name, body in files.items():
        path = raw / f"{name}.csv"
        path.write_text(body, encoding="utf-8")
        out[name] = path
    return out
