from navier_cfd import Catalog


def test_builtin_catalog_loads():
    catalog = Catalog.load_builtin()
    assert len(catalog.models) >= 50
    assert len(catalog.datasets) >= 10
    assert catalog.model("pibert").name == "PIBERT"
    assert catalog.dataset("cfdbench").hf_repo_id == "chen-yingfa/CFDBench"


def test_model_ids_are_unique():
    catalog = Catalog.load_builtin()
    ids = [m.id for m in catalog.models]
    assert len(ids) == len(set(ids))
