from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from navier_cfd import Catalog, load_cfd_dataset
from navier_cfd.datasets import (
    APEBenchDatasetManager,
    LOCAL_DATASET_CONTRACTS,
    LocalScientificDatasetManager,
)


def test_every_external_catalog_entry_has_a_runtime_provider() -> None:
    external_ids = {
        dataset.id
        for dataset in Catalog.load_builtin().datasets
        if dataset.provider == "external"
    }
    assert external_ids == set(LOCAL_DATASET_CONTRACTS) | {"apebench"}


def test_local_probe_reports_supported_formats(tmp_path) -> None:
    np.savez(
        tmp_path / "case.npz",
        coordinates=np.zeros((4, 3), dtype=np.float32),
        pressure=np.ones((4,), dtype=np.float32),
    )
    probe = LocalScientificDatasetManager().probe(
        "drivaernetpp",
        local_path=tmp_path,
    )
    assert probe.path_exists is True
    assert probe.supported_file_count == 1
    assert probe.formats[".npz"] == 1
    assert "pyvista" in probe.optional_dependencies


def test_drivaernetpp_local_point_cloud_adapter(tmp_path) -> None:
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    pressure = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    wall_shear_stress = np.ones((8, 3), dtype=np.float32)
    np.savez(
        tmp_path / "vehicle_001.npz",
        coordinates=coordinates,
        pressure=pressure,
        wall_shear_stress=wall_shear_stress,
    )

    dataset = load_cfd_dataset(
        "drivaernetpp",
        local_path=tmp_path,
        split="all",
        max_samples=1,
    )
    sample = dataset[0]
    assert sample.inputs.shape == (8, 3)
    assert sample.targets.shape == (8, 4)
    assert sample.coordinates.shape == (8, 3)
    assert sample.metadata["provider"] == "drivaernetpp_local"


def test_scalarflow_local_temporal_windows(tmp_path) -> None:
    density = np.arange(6 * 20 * 20, dtype=np.float32).reshape(6, 20, 20)
    velocity = np.ones((6, 20, 20, 2), dtype=np.float32)
    np.savez(tmp_path / "reconstruction.npz", density=density, velocity=velocity)

    dataset = load_cfd_dataset(
        "scalarflow",
        local_path=tmp_path,
        split="all",
        n_steps_input=2,
        n_steps_output=1,
        max_samples=1,
        max_windows=2,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.inputs.shape == (20, 20, 6)
    assert sample.targets.shape == (20, 20, 3)
    assert sample.coordinates.shape == (20, 20, 2)
    assert sample.metadata["provider"] == "scalarflow_local"


def test_local_provider_requires_explicit_local_path() -> None:
    with pytest.raises(ValueError, match="requires local_path"):
        load_cfd_dataset("eagle")


class _FakeScenario:
    def __init__(self, **kwargs):
        self.num_train_samples = int(kwargs.get("num_train_samples", 3))
        self.num_test_samples = int(kwargs.get("num_test_samples", 2))
        self.train_temporal_horizon = int(kwargs.get("train_temporal_horizon", 5))
        self.test_temporal_horizon = int(kwargs.get("test_temporal_horizon", 5))

    def _data(self, count: int, horizon: int) -> np.ndarray:
        return np.arange(count * horizon * 1 * 8, dtype=np.float32).reshape(
            count,
            horizon,
            1,
            8,
        )

    def get_train_data(self):
        return self._data(self.num_train_samples, self.train_temporal_horizon)

    def get_test_data(self):
        return self._data(self.num_test_samples, self.test_temporal_horizon)

    def get_scenario_name(self) -> str:
        return "fake_advection"


def test_apebench_official_api_adapter(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        __version__="test",
        scenarios=SimpleNamespace(
            difficulty=SimpleNamespace(Advection=_FakeScenario),
        ),
    )
    monkeypatch.setitem(sys.modules, "apebench", fake_module)

    dataset = APEBenchDatasetManager().load(
        "advection",
        split="train",
        n_steps_input=2,
        n_steps_output=1,
        max_samples=2,
        max_windows=2,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.inputs.shape == (8, 2)
    assert sample.targets.shape == (8, 1)
    assert sample.coordinates.shape == (8, 1)
    assert sample.metadata["provider"] == "apebench_procedural"
    assert dataset.access_plan["resolved_revision"] == "test"
