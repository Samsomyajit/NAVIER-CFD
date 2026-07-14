from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from navier_cfd import (
    Catalog,
    HuggingFaceDatasetManager,
    TheWellDatasetManager,
    load_cfd_dataset,
    load_model,
)


class FakeZScoreNormalization:
    pass


class FakeWellDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.use_normalization = kwargs.get("use_normalization", False)
        self.normalization_type = kwargs.get("normalization_type")
        self.metadata = SimpleNamespace(
            field_names={0: ["density"], 1: ["velocity_x", "velocity_y"]},
            constant_field_names={0: ["porosity"]},
            scalar_names=["forcing"],
            constant_scalar_names=["reynolds"],
            n_fields=3,
            n_spatial_dims=2,
            spatial_resolution=(4, 5),
            grid_type="cartesian",
            dataset_name=kwargs["well_dataset_name"],
        )
        x = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
        y = np.arange(1 * 4 * 5 * 3, dtype=np.float32).reshape(1, 4, 5, 3)
        self.records = [
            {
                "input_fields": x,
                "output_fields": y,
                "constant_fields": np.ones((4, 5, 1), dtype=np.float32),
                "input_scalars": np.array([[0.1], [0.2]], dtype=np.float32),
                "output_scalars": np.array([[0.3]], dtype=np.float32),
                "constant_scalars": np.array([100.0], dtype=np.float32),
                "boundary_conditions": np.array([[2, 2], [0, 1]], dtype=np.int64),
                "space_grid": (np.linspace(0, 1, 4), np.linspace(0, 1, 5)),
                "input_time_grid": np.array([0.0, 0.1]),
                "output_time_grid": np.array([0.2]),
            }
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def test_catalog_declares_the_well_as_provider_family() -> None:
    spec = Catalog.load_builtin().dataset("the_well")
    assert spec.hf_repo_id is None
    assert spec.provider == "the_well"
    assert spec.access_backend == "the_well.data.WellDataset"
    assert spec.access_base_path == "hf://datasets/polymathic-ai/"
    assert spec.requires_configuration is True
    assert spec.official_splits == ("train", "valid", "test")


def test_generic_huggingface_manager_rejects_the_well() -> None:
    spec = Catalog.load_builtin().dataset("the_well")
    with pytest.raises(ValueError, match="The Well is not one"):
        HuggingFaceDatasetManager().load(spec)


def test_official_provider_builds_and_adapts_records(monkeypatch) -> None:
    monkeypatch.setattr(TheWellDatasetManager, "_dataset_class", staticmethod(lambda: FakeWellDataset))
    monkeypatch.setattr(
        TheWellDatasetManager,
        "_default_normalization_type",
        staticmethod(lambda: FakeZScoreNormalization),
    )
    monkeypatch.setattr(TheWellDatasetManager, "provider_version", staticmethod(lambda: "test"))

    dataset = TheWellDatasetManager().load(
        "active_matter",
        split="train",
        streaming=True,
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=True,
    )
    sample = dataset[0]

    assert sample.inputs.shape == (4, 5, 7)
    assert sample.targets.shape == (4, 5, 3)
    assert sample.coordinates.shape == (4, 5, 2)
    assert sample.parameters["reynolds"] == pytest.approx(100.0)
    assert sample.metadata["well_dataset_name"] == "active_matter"
    assert sample.metadata["field_names"] == ("density", "velocity_x", "velocity_y")
    assert sample.metadata["input_channel_names"][-1] == "porosity"
    assert sample.metadata["normalization"]["type"] == "FakeZScoreNormalization"
    assert dataset.access_plan["base_path"] == "hf://datasets/polymathic-ai/"
    assert dataset.access_plan["normalization_type"] == "FakeZScoreNormalization"

    if importlib.util.find_spec("torch") is None:
        return
    model, plan = load_model(
        "fno",
        dataset="the_well",
        sample=sample,
        overrides={"modes": (2, 2), "width": 8, "n_layers": 1, "projection_width": 8},
        return_plan=True,
    )
    assert model.dimension == 2
    assert plan.builder_kwargs["in_channels"] == 7
    assert plan.builder_kwargs["out_channels"] == 3


def test_dataset_factory_requires_configuration() -> None:
    with pytest.raises(ValueError, match="requires configuration"):
        load_cfd_dataset("the_well")
