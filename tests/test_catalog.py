from navier_cfd import Catalog


def test_builtin_catalog_loads():
    catalog = Catalog.load_builtin()
    assert len(catalog.models) >= 50
    assert len(catalog.datasets) >= 10
    assert catalog.model("pibert").name == "PIBERT"
    cfdbench = catalog.dataset("cfdbench")
    assert cfdbench.provider == "cfdbench"
    assert cfdbench.hf_repo_id == "chen-yingfa/CFDBench-raw"


def test_model_ids_are_unique():
    catalog = Catalog.load_builtin()
    ids = [m.id for m in catalog.models]
    assert len(ids) == len(set(ids))
