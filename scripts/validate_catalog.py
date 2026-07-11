from navier_cfd import Catalog

catalog = Catalog.load_builtin()
print(f"models={len(catalog.models)} datasets={len(catalog.datasets)}")
for model in catalog.models:
    assert model.id and model.reference and model.categories
for dataset in catalog.datasets:
    assert dataset.id and dataset.description
print("catalog validation passed")
