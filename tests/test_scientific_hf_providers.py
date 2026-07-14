from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest

from navier_cfd import Catalog, load_cfd_dataset
from navier_cfd.datasets.auth import resolve_hf_auth
from navier_cfd.datasets.huggingface import classify_hf_layout
from navier_cfd.datasets.providers.cfdbench import load_cfdbench_archive_samples
from navier_cfd.datasets.providers.pdebench import PDEBenchHDF5Dataset
from navier_cfd.datasets.providers.realpdebench import RealPDEBenchTrajectoryDataset


def test_hf_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    assert resolve_hf_auth("explicit-token").source == "explicit"
    assert resolve_hf_auth("explicit-token").token == "explicit-token"
    assert resolve_hf_auth().source == "environment"
    assert resolve_hf_auth().token == "environment-token"
    assert resolve_hf_auth(False).source == "explicit_anonymous"
    assert resolve_hf_auth(False).token is None


def test_hf_layout_classification() -> None:
    assert classify_hf_layout(["data/file.h5"])[0] == "scientific_hdf5"
    assert classify_hf_layout(["case1.zip"])[0] == "archive"
    assert classify_hf_layout(["data-000.arrow", "state.json"])[0] == "saved_arrow"
    assert classify_hf_layout(["train.parquet"])[0] == "parquet"


def test_catalog_uses_provider_specific_scientific_loaders() -> None:
    catalog = Catalog.load_builtin()
    assert catalog.dataset("pdebench").provider == "pdebench"
    assert catalog.dataset("pdebench").hf_repo_id is None
    assert catalog.dataset("cfdbench").provider == "cfdbench"
    assert catalog.dataset("cfdbench").hf_repo_id == "chen-yingfa/CFDBench-raw"
    assert catalog.dataset("realpdebench").provider == "realpdebench"


def test_pdebench_hdf5_fixture(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "burgers.h5"
    tensor = np.arange(5 * 7 * 8, dtype=np.float32).reshape(5, 7, 8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
        handle.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, 8))
        handle.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, 7))

    dataset = PDEBenchHDF5Dataset(
        path,
        split="all",
        configuration="burgers",
        n_steps_input=2,
        n_steps_output=1,
        trajectory_limit=1,
        max_windows=2,
    )
    sample = dataset[0]
    assert sample.inputs.shape == (8, 2)
    assert sample.targets.shape == (8, 1)
    assert sample.coordinates.shape == (8, 1)
    assert sample.metadata["provider"] == "pdebench_hdf5"


def test_cfdbench_archive_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    header = "nodenumber x-coordinate y-coordinate x-velocity y-velocity absolute-pressure\n"
    for frame in range(3):
        rows = []
        for node in range(6):
            rows.append(
                f"{node} {node / 5:.3f} {frame / 2:.3f} "
                f"{node + frame:.3f} {node - frame:.3f} {10 + node:.3f}\n"
            )
        (source / f"frame_{frame:03d}.txt").write_text(header + "".join(rows), encoding="utf-8")
    archive = tmp_path / "case2.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for file in source.iterdir():
            handle.write(file, file.name)

    samples = load_cfdbench_archive_samples(
        archive,
        scenario="cavity",
        extract_dir=tmp_path / "extracted",
    )
    assert len(samples) == 3
    assert samples[0].coordinates.shape == (6, 2)
    assert samples[0].targets.shape == (6, 3)
    assert samples[0].metadata["provider"] == "cfdbench_archive"


def _realpde_row(index: int) -> dict[str, object]:
    shape = (7, 3, 2)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, shape[2], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
    )
    return {
        "sim_id": f"case-{index}.h5",
        "u": base.tobytes(),
        "v": (base * 0.5).tobytes(),
        "x": x.tobytes(),
        "y": y.tobytes(),
        "t": np.linspace(0.0, 1.0, shape[0], dtype=np.float32).tobytes(),
        "shape_t": shape[0],
        "shape_h": shape[1],
        "shape_w": shape[2],
    }


def test_realpdebench_arrow_row_fixture() -> None:
    dataset = RealPDEBenchTrajectoryDataset(
        [_realpde_row(0), _realpde_row(1)],
        scenario="fsi",
        data_type="real",
        split="all",
        n_steps_input=2,
        n_steps_output=1,
        window_stride=2,
        max_windows=2,
    )
    sample = dataset[0]
    assert sample.inputs.shape == (3, 2, 4)
    assert sample.targets.shape == (3, 2, 2)
    assert sample.coordinates.shape == (3, 2, 2)
    assert sample.metadata["subset_mode"] is True
    assert sample.metadata["official_split"] is False


def test_factory_routes_pdebench_to_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    import navier_cfd.datasets.factory as factory

    class DummyManager:
        def __init__(self, token=None, endpoint=None):
            self.token = token
            self.endpoint = endpoint

        def load(self, configuration, **kwargs):
            return {"configuration": configuration, **kwargs}

    monkeypatch.setattr(factory, "PDEBenchDatasetManager", DummyManager)
    result = load_cfd_dataset(
        "pdebench",
        configuration="burgers",
        split="test",
        token="secret",
        filename="tiny.h5",
    )
    assert result["configuration"] == "burgers"
    assert result["split"] == "test"
    assert result["filename"] == "tiny.h5"
