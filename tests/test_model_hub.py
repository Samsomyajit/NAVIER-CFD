from __future__ import annotations

import importlib.util

import pytest

from navier_cfd import ModelHub, TaskSpec, list_models, model_info
from navier_cfd.models import ModelDependencyError, ModelNotExecutableError


def _task(dimension: int = 2) -> TaskSpec:
    return TaskSpec(
        problem="test_flow",
        task_type="surrogate",
        dimension=dimension,
        mesh_type="structured",
        temporal_mode="steady",
        geometry_mode="fixed",
        physics=("fluid_dynamics",),
    )


def test_every_catalog_model_has_a_uniform_handle() -> None:
    handles = list_models()
    assert len(handles) >= 55
    assert len({handle.id for handle in handles}) == len(handles)
    assert all(handle.status.model_id == handle.id for handle in handles)


def test_native_models_are_reported_as_executable() -> None:
    for model_id in ("pinn", "deeponet", "fno"):
        runtime = model_info(model_id)["runtime"]
        assert runtime["mode"] == "native"
        assert runtime["executable"] is True
        assert runtime["installable"] is True


def test_external_entrypoint_can_be_connected_without_package_fork() -> None:
    hub = ModelHub()
    hub.register_external("transolver", entrypoint="collections:Counter")
    result = hub.load("transolver", "abca")
    assert result["a"] == 2
    assert result["b"] == 1


def test_metadata_only_model_explains_how_to_connect_adapter() -> None:
    hub = ModelHub()
    with pytest.raises(ModelNotExecutableError, match="register_external|entrypoint"):
        hub.load("mionet", task=_task())


def test_native_models_forward_when_torch_is_installed() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is an optional dependency")
    import torch

    hub = ModelHub()

    fno = hub.load(
        "fno",
        task=_task(2),
        in_channels=3,
        out_channels=2,
        modes=(4, 4),
        width=8,
        n_layers=2,
        projection_width=16,
    )
    fno_inputs = torch.randn(2, 12, 10, 3)
    assert fno(fno_inputs).shape == (2, 12, 10, 2)

    pinn = hub.load(
        "pinn",
        task=_task(2),
        input_dim=3,
        output_dim=3,
        hidden_channels=16,
        depth=2,
    )
    assert pinn(torch.randn(4, 3)).shape == (4, 3)

    deeponet = hub.load(
        "deeponet",
        task=_task(2),
        branch_input_dim=8,
        trunk_input_dim=2,
        output_dim=2,
        latent_dim=8,
        hidden_channels=16,
        depth=2,
    )
    branch = torch.randn(3, 8)
    trunk = torch.randn(5, 2)
    assert deeponet(branch, trunk).shape == (3, 5, 2)


def test_native_load_has_actionable_error_without_torch() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("This test only applies when PyTorch is absent")
    hub = ModelHub()
    with pytest.raises(ModelDependencyError, match=r"navier-cfd\[torch\]"):
        hub.load("pinn", task=_task(), output_dim=2)
