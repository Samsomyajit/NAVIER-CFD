from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from navier_cfd import (
    CFDSample,
    DATASET_MODEL_DEFAULTS,
    ModelHub,
    configure_model_for_dataset,
    load_model,
)
from navier_cfd.datasets import collate_cfd_samples
from navier_cfd.models.forward import forward_model

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is optional")


NATIVE_SUITE = (
    "pinn",
    "nsfnets",
    "pinnsformer",
    "deeponet",
    "mionet",
    "fourier_deeponet",
    "nested_fourier_deeponet",
    "fourier_mionet",
    "fno",
    "pino",
    "geo_fno",
    "gino",
    "u_fno",
    "f_fno",
    "u_no",
    "lsm",
    "gnot",
    "galerkin_transformer",
    "mwt",
    "factformer",
    "ono",
    "transolver",
    "upt",
    "meshgraphnets",
    "domino",
    "pibert",
    "fourierflow",
    "pde_refiner",
    "dpot",
    "poseidon",
    "prose_fd",
    "bcat",
    "pdeformer1",
    "pi_mfm",
    "laplace_no",
    "state_space_no",
    "p3d",
    "aerotransformer",
    "tadpole",
    "solver_in_loop",
    "inc",
    "neurosem",
    "np_newton",
    "geometry_preconditioner",
    "revit",
    "deepmmnet",
    "conformal_deeponet",
    "tante",
    "riemannonet",
    "energy_transformer",
    "fun_diff",
    "flow_matching_pde",
)


def _sample() -> CFDSample:
    grid = np.stack(
        np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), indexing="ij"),
        axis=-1,
    ).astype(np.float32)
    inputs = np.concatenate((grid, grid[..., :1] ** 2), axis=-1).astype(np.float32)
    targets = np.stack((inputs[..., 0] + inputs[..., 1], inputs[..., 0] - inputs[..., 1]), axis=-1)
    return CFDSample(inputs=inputs, targets=targets.astype(np.float32), coordinates=grid)


SMALL_OVERRIDES = {
    "hidden_dim": 16,
    "hidden_channels": 16,
    "width": 8,
    "latent_width": 12,
    "latent_dim": 8,
    "num_layers": 1,
    "n_layers": 1,
    "depth": 2,
    "num_heads": 2,
    "k_neighbors": 3,
    "modes": (2, 2),
    "num_frequencies": 2,
    "wavelet_scales": (1,),
}


def test_all_dataset_profiles_resolve_without_a_sample() -> None:
    for dataset_id, defaults in DATASET_MODEL_DEFAULTS.items():
        plan = configure_model_for_dataset("pibert", dataset_id)
        assert plan.dataset_id == dataset_id
        assert plan.dataset_configuration["dimension"] == defaults.dimension
        assert plan.builder_kwargs["input_dim"] == defaults.input_channels
        assert plan.builder_kwargs["output_dim"] == defaults.output_channels


def test_native_suite_is_importable_and_backward_tested() -> None:
    sample = _sample()
    batch = collate_cfd_samples([sample])
    hub = ModelHub()
    failures: list[str] = []

    for model_id in NATIVE_SUITE:
        status = hub.status(model_id)
        if status.mode != "native" or not status.executable:
            failures.append(f"{model_id}: status={status.mode}/{status.executable}")
            continue
        try:
            model = load_model(
                model_id,
                dataset="pdebench",
                sample=sample,
                overrides=SMALL_OVERRIDES,
                hub=hub,
            )
            output = forward_model(model_id, model, batch)
            assert output.numel() == batch.targets.numel()
            output.reshape_as(batch.targets).square().mean().backward()
            assert any(parameter.grad is not None for parameter in model.parameters())
            assert model.navier_dataset_id == "pdebench"
        except Exception as exc:  # pragma: no cover - collected for actionable CI output
            failures.append(f"{model_id}: {type(exc).__name__}: {exc}")

    assert len(NATIVE_SUITE) == 52
    assert not failures, "\n".join(failures)


def test_dataset_argument_changes_model_configuration() -> None:
    fno_2d, plan_2d = load_model(
        "fno",
        dataset="cfdbench",
        overrides={"modes": (2, 2), "width": 8, "n_layers": 1, "projection_width": 8},
        return_plan=True,
    )
    fno_3d, plan_3d = load_model(
        "fno",
        dataset="scalarflow",
        overrides={"modes": (2, 2, 2), "width": 8, "n_layers": 1, "projection_width": 8},
        return_plan=True,
    )
    assert fno_2d.dimension == 2
    assert fno_3d.dimension == 3
    assert plan_2d.builder_kwargs["in_channels"] == 3
    assert plan_3d.builder_kwargs["in_channels"] == 1
    assert plan_2d.dataset_configuration["representation"] == "structured"


def test_point_cloud_dataset_selects_coordinate_conditioned_configuration() -> None:
    model, plan = load_model(
        "transolver",
        dataset="airfrans",
        overrides={"hidden_dim": 16, "num_layers": 1, "num_heads": 2},
        return_plan=True,
    )
    assert model.navier_input_mode == "field_with_coordinates"
    assert plan.dataset_configuration["representation"] == "point_cloud"
    assert plan.builder_kwargs["coordinate_dim"] == 2
