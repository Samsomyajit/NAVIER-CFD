from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from navier_cfd import (
    AdaptedDataset,
    AdapterRegistry,
    CFDTrainer,
    CheckpointManager,
    ModelHub,
    TrainerConfig,
    make_dataloaders,
)

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is optional")


def _records(count: int = 8) -> list[dict[str, np.ndarray]]:
    grid = np.stack(
        np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2), indexing="ij"),
        axis=-1,
    ).astype(np.float32)
    rows = []
    for index in range(count):
        inputs = np.empty((2, 2, 2), dtype=np.float32)
        inputs[..., 0] = grid[..., 0] + 0.01 * index
        inputs[..., 1] = grid[..., 1]
        targets = np.stack(
            (inputs[..., 0] + inputs[..., 1], inputs[..., 0] - inputs[..., 1]),
            axis=-1,
        ).astype(np.float32)
        rows.append({"input": inputs, "target": targets, "coordinates": grid})
    return rows


def test_pibert_is_native_and_supports_structured_and_point_sequences() -> None:
    import torch

    hub = ModelHub()
    status = hub.status("pibert")
    assert status.mode == "native"
    assert status.executable is True

    model = hub.load(
        "pibert",
        input_dim=3,
        output_dim=2,
        coordinate_dim=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        num_frequencies=2,
        wavelet_scales=(1,),
    )
    structured = torch.randn(2, 3, 2, 3)
    coordinates = torch.randn(2, 3, 2, 2)
    assert model(structured, coordinates=coordinates).shape == (2, 3, 2, 2)

    points = torch.randn(2, 5, 3)
    point_coordinates = torch.randn(2, 5, 2)
    assert model(points, coordinates=point_coordinates).shape == (2, 5, 2)


def test_unified_training_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    dataset = AdaptedDataset(_records(), AdapterRegistry().adapter("pdebench"))
    loaders = make_dataloaders(
        dataset,
        batch_size=2,
        train=0.5,
        validation=0.25,
        test=0.25,
        seed=3,
    )
    model = ModelHub().load(
        "pibert",
        input_dim=2,
        output_dim=2,
        coordinate_dim=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        num_frequencies=2,
        wavelet_scales=(1,),
    )
    trainer = CFDTrainer(
        model,
        model_id="pibert",
        config=TrainerConfig(
            epochs=1,
            mixed_precision=False,
            scheduler=None,
            checkpoint_dir=str(tmp_path / "training"),
        ),
    )
    result = trainer.fit(loaders["train"], loaders["validation"])
    assert result.best_epoch == 1
    metrics = trainer.evaluate(loaders["test"])
    assert {"rmse", "relative_l2", "r2", "spectral_relative_error"} <= metrics.keys()

    manual = tmp_path / "manual"
    manager = CheckpointManager()
    manager.save(
        manual,
        model=model,
        optimizer=trainer.optimizer,
        config=trainer.config,
        metrics=metrics,
        metadata={"model_id": "pibert", "dataset_id": "pdebench"},
        epoch=1,
    )
    manifest = manager.load(manual, model=model, optimizer=trainer.optimizer)
    assert manifest["schema"] == "navier-cfd.checkpoint/v1"
    assert manifest["metadata"]["model_id"] == "pibert"
