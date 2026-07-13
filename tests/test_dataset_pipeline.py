from __future__ import annotations

import numpy as np

from navier_cfd import AdaptedDataset, AdapterRegistry, CFDSample, split_indices, translate_model_config


def test_all_registered_datasets_have_adapter_profiles() -> None:
    registry = AdapterRegistry()
    assert set(registry.ids()) == {
        "airfrans",
        "apebench",
        "cfdbench",
        "drivaerml",
        "drivaernetpp",
        "eagle",
        "pdebench",
        "realpdebench",
        "scalarflow",
        "shapenet_car",
        "the_well",
    }


def test_structured_record_adapts_to_canonical_sample() -> None:
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3), indexing="ij"), axis=-1)
    record = {
        "input": np.ones((4, 3, 2)),
        "target": np.zeros((4, 3, 3)),
        "coordinates": grid,
        "parameters": {"reynolds": 100.0},
        "case": "cylinder",
    }
    sample = AdapterRegistry().adapter("pdebench").adapt(record)
    assert sample.inputs.shape == (4, 3, 2)
    assert sample.targets.shape == (4, 3, 3)
    assert sample.coordinates.shape == (4, 3, 2)
    assert sample.parameters["reynolds"] == 100.0
    assert sample.metadata["case"] == "cylinder"


def test_point_cloud_aliases_and_field_concatenation() -> None:
    record = {
        "node_features": {"nx": np.ones(5), "ny": np.zeros(5)},
        "fields": {"pressure": np.arange(5), "wall_shear": np.ones(5)},
        "pos": np.zeros((5, 3)),
    }
    sample = AdapterRegistry().adapter(
        "airfrans",
        input_fields=("nx", "ny"),
        target_fields=("pressure", "wall_shear"),
    ).adapt(record)
    assert sample.inputs.shape == (5, 2)
    assert sample.targets.shape == (5, 2)
    assert sample.coordinates.shape == (5, 3)


def test_adapted_dataset_and_reproducible_splits() -> None:
    raw = [{"x": np.ones((3, 1)), "y": np.zeros((3, 1))} for _ in range(10)]
    dataset = AdaptedDataset(raw, AdapterRegistry().adapter("pdebench"))
    assert isinstance(dataset[0], CFDSample)
    first = split_indices(10, seed=17)
    second = split_indices(10, seed=17)
    assert first == second
    assert sorted(first["train"] + first["validation"] + first["test"]) == list(range(10))


def test_model_configuration_translation_for_pibert_and_fno() -> None:
    sample = CFDSample(
        inputs=np.zeros((8, 6, 4), dtype=np.float32),
        targets=np.zeros((8, 6, 3), dtype=np.float32),
        coordinates=np.zeros((8, 6, 2), dtype=np.float32),
    )
    pibert = translate_model_config("pibert", sample)
    assert pibert.builder_kwargs["input_dim"] == 4
    assert pibert.builder_kwargs["output_dim"] == 3
    assert pibert.builder_kwargs["coordinate_dim"] == 2
    assert pibert.input_mode == "field_with_coordinates"

    fno = translate_model_config("fno", sample)
    assert fno.builder_kwargs["in_channels"] == 4
    assert fno.builder_kwargs["out_channels"] == 3
