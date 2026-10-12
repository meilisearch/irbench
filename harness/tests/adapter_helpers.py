from conftest import REPO_ROOT
from irbench.engines.base import IndexContext, load_adapter_class
from irbench.engines.config import TrackConfig
from irbench.models import QueryTask
from irbench.scenario import load_scenario
from irbench.terms import load_term_groups

SCENARIO = "ecommerce/product-search"


def make_adapter(engine, track=None, dim=3, n_docs=None):
    cls = load_adapter_class(REPO_ROOT, engine)
    scenario = load_scenario(REPO_ROOT, SCENARIO)
    return cls(
        IndexContext(
            scenario=scenario,
            config=TrackConfig.model_validate(track or {}),
            index="idx",
            dim=dim,
            synonyms=load_term_groups(scenario.synonyms_path),
            n_docs=n_docs,
        )
    )


def make_task(mode, base=None, text="grey sofa", qid="1", **extra):
    base = base or mode
    return QueryTask(
        task_id=f"{qid}:{mode}:{base}",
        qid=qid,
        variant_of=qid,
        perturbation="original",
        text=text,
        mode=mode,
        base=base,
        **extra,
    )
