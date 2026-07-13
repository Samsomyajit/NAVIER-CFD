from __future__ import annotations

from typing import Any, Callable

from .native import MissingTorchDependency


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingTorchDependency(
            "Native NAVIER-CFD models require PyTorch. Install `navier-cfd[models]`."
        ) from exc
    return nn, functional, functional


def build_latent_reference(
    model_id: str,
    *,
    input_dim: int,
    output_dim: int,
    dimension: int = 2,
    width: int = 64,
    latent_width: int = 128,
    num_layers: int = 3,
    **_: Any,
) -> Any:
    """Build a dimension-generic latent residual operator.

    The operator downsamples a structured field into a latent grid, applies
    residual convolutional blocks, and interpolates back to the original grid.
    """

    nn, functional, _ = _require_torch()
    if dimension not in {1, 2, 3}:
        raise ValueError("dimension must be 1, 2, or 3")
    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimension]
    pool = {1: functional.avg_pool1d, 2: functional.avg_pool2d, 3: functional.avg_pool3d}[dimension]
    interpolation_mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[dimension]

    class LatentReferenceOperator(nn.Module):
        navier_input_mode = "field"

        def __init__(self) -> None:
            super().__init__()
            self.lift = conv(input_dim, width, 1)
            self.encoder = nn.Sequential(
                conv(width, latent_width, 3, padding=1),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        conv(latent_width, latent_width, 3, padding=1),
                        nn.GELU(),
                        conv(latent_width, latent_width, 1),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.output = conv(latent_width, output_dim, 1)
            self.navier_reference_model_id = model_id
            self.navier_reference_notice = (
                "NAVIER-CFD native reference implementation; validate against the official paper and repository "
                "before claiming reproduction."
            )

        def forward(self, values: Any, **__: Any) -> Any:
            original_size = tuple(values.shape[1:-1])
            hidden = self.encoder(self.lift(values.movedim(-1, 1)))
            if min(hidden.shape[2:]) >= 2:
                hidden = pool(hidden, kernel_size=2, stride=2)
            for block in self.blocks:
                hidden = hidden + block(hidden)
            hidden = functional.interpolate(
                hidden,
                size=original_size,
                mode=interpolation_mode,
                align_corners=False,
            )
            return self.output(hidden).movedim(1, -1)

    return LatentReferenceOperator()


def _builder(model_id: str, **defaults: Any) -> Callable[..., Any]:
    def build(*, task: Any | None = None, spec: Any | None = None, **kwargs: Any) -> Any:
        del task, spec
        return build_latent_reference(model_id, **{**defaults, **kwargs})

    build.__name__ = f"build_{model_id}"
    return build


LATENT_BUILDERS = {
    "u_fno": _builder("u_fno", num_layers=3),
    "u_no": _builder("u_no", num_layers=4),
    "lsm": _builder("lsm", latent_width=160),
    "tadpole": _builder("tadpole", num_layers=5),
}


__all__ = ["LATENT_BUILDERS", "build_latent_reference"]
