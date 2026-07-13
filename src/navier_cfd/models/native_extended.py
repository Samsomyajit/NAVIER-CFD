from __future__ import annotations

from typing import Any, Callable

from .native import MissingTorchDependency
from .native_suite import ConvOperator, GraphOperator, PointTransformerOperator


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingTorchDependency(
            "Native NAVIER-CFD models require PyTorch. Install `navier-cfd[models]`."
        ) from exc
    return torch, nn


def _tag(model: Any, model_id: str, capability: str) -> Any:
    model.navier_reference_model_id = model_id
    model.navier_reference_capability = capability
    model.navier_reference_notice = (
        "NAVIER-CFD native reference implementation; validate architecture, preprocessing, losses, and checkpoints "
        "against the official paper and repository before claiming reproduction."
    )
    return model


def build_corrector_reference(
    model_id: str,
    *,
    input_dim: int,
    output_dim: int,
    dimension: int = 2,
    width: int = 64,
    num_layers: int = 4,
    residual_scale: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Build a learned field corrector for solver-in-loop and preconditioner studies."""

    _, nn = _require_torch()
    backbone = ConvOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        dimension=dimension,
        width=width,
        num_layers=num_layers,
        multiscale=True,
        **kwargs,
    )

    class CorrectorReference(nn.Module):
        navier_input_mode = "field"

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.residual_scale = float(residual_scale)

        def forward(self, values: Any, **forward_kwargs: Any) -> Any:
            correction = self.backbone(values, **forward_kwargs)
            if values.shape[-1] == correction.shape[-1]:
                return values + self.residual_scale * correction
            return correction

        def correct(self, state: Any, **forward_kwargs: Any) -> Any:
            return self(state, **forward_kwargs)

    return _tag(CorrectorReference(), model_id, "learned_corrector")


def build_generative_reference(
    model_id: str,
    *,
    input_dim: int,
    output_dim: int,
    dimension: int = 2,
    width: int = 64,
    num_layers: int = 5,
    noise_channels: int = 1,
    **kwargs: Any,
) -> Any:
    """Build a conditional stochastic field refiner with deterministic training output."""

    torch, nn = _require_torch()
    backbone = ConvOperator(
        input_dim=input_dim + noise_channels,
        output_dim=output_dim,
        dimension=dimension,
        width=width,
        num_layers=num_layers,
        multiscale=True,
        **kwargs,
    )

    class GenerativeReference(nn.Module):
        navier_input_mode = "field"

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.noise_channels = noise_channels
            self.log_noise_scale = nn.Parameter(torch.full((noise_channels,), -3.0))

        def _condition(self, values: Any, noise: Any | None = None) -> Any:
            if noise is None:
                shape = (*values.shape[:-1], self.noise_channels)
                noise = torch.zeros(shape, dtype=values.dtype, device=values.device)
            scale = self.log_noise_scale.exp()
            return torch.cat((values, noise * scale), dim=-1)

        def forward(self, values: Any, noise: Any | None = None, **forward_kwargs: Any) -> Any:
            return self.backbone(self._condition(values, noise), **forward_kwargs)

        def sample(self, values: Any, samples: int = 1) -> Any:
            outputs = []
            for _ in range(samples):
                noise = torch.randn(*values.shape[:-1], self.noise_channels, device=values.device, dtype=values.dtype)
                outputs.append(self(values, noise=noise))
            return torch.stack(outputs, dim=1)

    return _tag(GenerativeReference(), model_id, "conditional_generative_refiner")


def build_conformal_reference(
    model_id: str,
    *,
    input_dim: int,
    output_dim: int,
    coordinate_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    **kwargs: Any,
) -> Any:
    """Build a coordinate-aware predictor with split-conformal interval support."""

    torch, nn = _require_torch()
    predictor = PointTransformerOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        coordinate_dim=coordinate_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )

    class ConformalReference(nn.Module):
        navier_input_mode = "field_with_coordinates"

        def __init__(self) -> None:
            super().__init__()
            self.predictor = predictor
            self.register_buffer("calibration_radius", torch.zeros(output_dim))

        def forward(self, values: Any, coordinates: Any | None = None, mask: Any | None = None, **forward_kwargs: Any) -> Any:
            return self.predictor(values, coordinates=coordinates, mask=mask, **forward_kwargs)

        def calibrate(self, prediction: Any, target: Any, coverage: float = 0.9) -> Any:
            if not 0.0 < coverage < 1.0:
                raise ValueError("coverage must lie between 0 and 1")
            residual = (target - prediction).abs().reshape(-1, target.shape[-1])
            self.calibration_radius.copy_(torch.quantile(residual, coverage, dim=0))
            return self.calibration_radius

        def predict_interval(self, values: Any, coordinates: Any | None = None, mask: Any | None = None) -> tuple[Any, Any, Any]:
            center = self(values, coordinates=coordinates, mask=mask)
            return center, center - self.calibration_radius, center + self.calibration_radius

    return _tag(ConformalReference(), model_id, "conformal_uncertainty")


def build_adaptive_reference(
    model_id: str,
    *,
    input_dim: int,
    output_dim: int,
    coordinate_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    **kwargs: Any,
) -> Any:
    """Build a geometry-aware predictor with explicit test-time adaptation hooks."""

    torch, nn = _require_torch()
    predictor = PointTransformerOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        coordinate_dim=coordinate_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs,
    )

    class AdaptiveReference(nn.Module):
        navier_input_mode = "field_with_coordinates"

        def __init__(self) -> None:
            super().__init__()
            self.predictor = predictor

        def forward(self, values: Any, coordinates: Any | None = None, mask: Any | None = None, **forward_kwargs: Any) -> Any:
            return self.predictor(values, coordinates=coordinates, mask=mask, **forward_kwargs)

        def adapt(self, values: Any, target: Any, coordinates: Any | None = None, steps: int = 1, learning_rate: float = 1e-4) -> float:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            loss_value = 0.0
            self.train()
            for _ in range(max(1, steps)):
                optimizer.zero_grad(set_to_none=True)
                prediction = self(values, coordinates=coordinates)
                loss = (prediction - target).square().mean()
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach())
            return loss_value

    return _tag(AdaptiveReference(), model_id, "test_time_adaptation")


def build_geometry_reference(
    model_id: str,
    *,
    input_dim: int,
    output_dim: int,
    coordinate_dim: int = 3,
    hidden_dim: int = 128,
    num_layers: int = 5,
    k_neighbors: int = 16,
    **kwargs: Any,
) -> Any:
    model = GraphOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        coordinate_dim=coordinate_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        k_neighbors=k_neighbors,
        **kwargs,
    )
    return _tag(model, model_id, "geometry_graph_operator")


def _builder(factory: Callable[..., Any], model_id: str, **defaults: Any) -> Callable[..., Any]:
    def build(*, task: Any | None = None, spec: Any | None = None, **kwargs: Any) -> Any:
        del task, spec
        return factory(model_id, **{**defaults, **kwargs})

    build.__name__ = f"build_{model_id}"
    return build


EXTENDED_BUILDERS = {
    "domino": _builder(build_geometry_reference, "domino", num_layers=6),
    "fourierflow": _builder(build_generative_reference, "fourierflow", num_layers=5),
    "pde_refiner": _builder(build_generative_reference, "pde_refiner", num_layers=6),
    "solver_in_loop": _builder(build_corrector_reference, "solver_in_loop", num_layers=4),
    "inc": _builder(build_corrector_reference, "inc", num_layers=5),
    "neurosem": _builder(build_corrector_reference, "neurosem", num_layers=5),
    "np_newton": _builder(build_corrector_reference, "np_newton", num_layers=4),
    "geometry_preconditioner": _builder(build_geometry_reference, "geometry_preconditioner", num_layers=4),
    "conformal_deeponet": _builder(build_conformal_reference, "conformal_deeponet", num_layers=4),
    "tante": _builder(build_adaptive_reference, "tante", num_layers=4),
    "energy_transformer": _builder(build_adaptive_reference, "energy_transformer", num_layers=5),
    "fun_diff": _builder(build_generative_reference, "fun_diff", num_layers=6),
    "flow_matching_pde": _builder(build_generative_reference, "flow_matching_pde", num_layers=6),
}


__all__ = [
    "EXTENDED_BUILDERS",
    "build_adaptive_reference",
    "build_conformal_reference",
    "build_corrector_reference",
    "build_generative_reference",
    "build_geometry_reference",
]
