from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Sequence


class MissingTorchDependency(ImportError):
    """Raised when an executable native model is requested without PyTorch."""


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise MissingTorchDependency(
            "Executable NAVIER-CFD models require PyTorch. Install them with "
            "`pip install 'navier-cfd[torch]'`."
        ) from exc
    return torch, nn


def _activation(nn: Any, name: str) -> Any:
    key = name.strip().lower()
    table = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if key not in table:
        raise ValueError(f"Unsupported activation {name!r}; choose one of {sorted(table)}")
    return table[key]


def _mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_channels: int,
    depth: int,
    activation: str,
) -> Any:
    _, nn = _require_torch()
    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    if hidden_channels < 1 or depth < 1:
        raise ValueError("hidden_channels and depth must be positive")

    act = _activation(nn, activation)
    layers: list[Any] = []
    width_in = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width_in, hidden_channels), act()))
        width_in = hidden_channels
    layers.append(nn.Linear(width_in, output_dim))
    return nn.Sequential(*layers)


def build_pinn(
    *,
    task: Any | None = None,
    input_dim: int | None = None,
    output_dim: int = 1,
    hidden_channels: int = 128,
    depth: int = 4,
    activation: str = "tanh",
    **_: Any,
) -> Any:
    """Build a coordinate-network PINN backbone.

    The returned module maps coordinates/parameters to predicted state variables.
    PDE residuals and boundary losses remain user-defined because they are specific
    to the governing equations and discretization.
    """

    inferred = getattr(task, "dimension", None)
    return _mlp(
        input_dim=input_dim or inferred or 2,
        output_dim=output_dim,
        hidden_channels=hidden_channels,
        depth=depth,
        activation=activation,
    )


def build_deeponet(
    *,
    task: Any | None = None,
    branch_input_dim: int,
    trunk_input_dim: int | None = None,
    output_dim: int = 1,
    latent_dim: int = 128,
    hidden_channels: int = 128,
    depth: int = 3,
    activation: str = "gelu",
    bias: bool = True,
    **_: Any,
) -> Any:
    """Build a DeepONet with branch and trunk networks.

    Forward accepts ``branch_input`` of shape ``[B, branch_input_dim]`` and
    ``trunk_input`` of shape ``[N, trunk_input_dim]`` or
    ``[B, N, trunk_input_dim]``. The output shape is ``[B, N, output_dim]``.
    """

    torch, nn = _require_torch()
    trunk_dim = trunk_input_dim or getattr(task, "dimension", None) or 2
    if latent_dim < 1 or output_dim < 1:
        raise ValueError("latent_dim and output_dim must be positive")

    branch = _mlp(
        branch_input_dim,
        latent_dim * output_dim,
        hidden_channels=hidden_channels,
        depth=depth,
        activation=activation,
    )
    trunk = _mlp(
        trunk_dim,
        latent_dim * output_dim,
        hidden_channels=hidden_channels,
        depth=depth,
        activation=activation,
    )

    class DeepONet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.branch = branch
            self.trunk = trunk
            self.output_dim = output_dim
            self.latent_dim = latent_dim
            self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

        def forward(self, branch_input: Any, trunk_input: Any) -> Any:
            branch_features = self.branch(branch_input).reshape(
                branch_input.shape[0], self.output_dim, self.latent_dim
            )
            trunk_features = self.trunk(trunk_input)
            if trunk_input.ndim == 2:
                trunk_features = trunk_features.reshape(
                    trunk_input.shape[0], self.output_dim, self.latent_dim
                )
                output = torch.einsum("bol,nol->bno", branch_features, trunk_features)
            elif trunk_input.ndim == 3:
                trunk_features = trunk_features.reshape(
                    trunk_input.shape[0], trunk_input.shape[1], self.output_dim, self.latent_dim
                )
                output = torch.einsum("bol,bnol->bno", branch_features, trunk_features)
            else:
                raise ValueError("trunk_input must have shape [N,D] or [B,N,D]")
            if self.bias is not None:
                output = output + self.bias
            return output

    return DeepONet()


def _normalize_modes(modes: int | Sequence[int], dimension: int) -> tuple[int, ...]:
    if isinstance(modes, int):
        values = (modes,) * dimension
    else:
        values = tuple(int(value) for value in modes)
    if len(values) != dimension or any(value < 1 for value in values):
        raise ValueError(f"modes must contain {dimension} positive integers")
    return values


def build_fno(
    *,
    task: Any | None = None,
    dimension: int | None = None,
    in_channels: int,
    out_channels: int,
    modes: int | Sequence[int] = 16,
    width: int = 64,
    n_layers: int = 4,
    projection_width: int = 128,
    activation: str = "gelu",
    channel_last: bool = True,
    **_: Any,
) -> Any:
    """Build a dimension-generic Fourier Neural Operator reference model.

    Supported dimensions are 1D, 2D and 3D. Inputs are channel-last by default:
    ``[B, X, C]``, ``[B, X, Y, C]`` or ``[B, X, Y, Z, C]``. Set
    ``channel_last=False`` for standard PyTorch channel-first tensors.
    """

    torch, nn = _require_torch()
    dim = int(dimension or getattr(task, "dimension", 0) or 2)
    if dim not in {1, 2, 3}:
        raise ValueError("FNO dimension must be 1, 2, or 3")
    if min(in_channels, out_channels, width, n_layers, projection_width) < 1:
        raise ValueError("channel counts, width, layers and projection width must be positive")

    mode_tuple = _normalize_modes(modes, dim)
    conv_type = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
    act_type = _activation(nn, activation)

    class SpectralConvND(nn.Module):
        def __init__(self, channels_in: int, channels_out: int) -> None:
            super().__init__()
            self.channels_in = channels_in
            self.channels_out = channels_out
            self.modes = mode_tuple
            block_count = 2 ** max(dim - 1, 0)
            scale = 1.0 / max(1, channels_in * channels_out)
            shape = (channels_in, channels_out, *mode_tuple)
            self.weights = nn.ParameterList(
                [nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(block_count)]
            )

        def forward(self, value: Any) -> Any:
            spatial_dims = tuple(range(2, 2 + dim))
            transformed = torch.fft.rfftn(value, dim=spatial_dims, norm="ortho")
            output_shape = (value.shape[0], self.channels_out, *transformed.shape[2:])
            output = torch.zeros(output_shape, dtype=transformed.dtype, device=value.device)
            actual_modes = tuple(min(self.modes[i], transformed.shape[2 + i]) for i in range(dim))

            sign_patterns: Iterable[tuple[int, ...]]
            sign_patterns = product((0, 1), repeat=max(dim - 1, 0))
            for weight, signs in zip(self.weights, sign_patterns):
                data_slices: list[slice] = [slice(None), slice(None)]
                for axis, count in enumerate(actual_modes):
                    if axis < dim - 1 and signs[axis] == 1:
                        data_slices.append(slice(-count, None))
                    else:
                        data_slices.append(slice(0, count))
                weight_slices = (slice(None), slice(None), *[slice(0, count) for count in actual_modes])
                block = transformed[tuple(data_slices)]
                output[tuple(data_slices)] = torch.einsum(
                    "bi...,io...->bo...", block, weight[weight_slices]
                )
            return torch.fft.irfftn(output, s=value.shape[2:], dim=spatial_dims, norm="ortho")

    class FourierNeuralOperator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dimension = dim
            self.channel_last = channel_last
            self.lift = nn.Linear(in_channels, width)
            self.spectral_layers = nn.ModuleList([SpectralConvND(width, width) for _ in range(n_layers)])
            self.local_layers = nn.ModuleList([conv_type(width, width, kernel_size=1) for _ in range(n_layers)])
            self.activation = act_type()
            self.projection = nn.Sequential(
                nn.Linear(width, projection_width),
                act_type(),
                nn.Linear(projection_width, out_channels),
            )

        def forward(self, value: Any) -> Any:
            expected_ndim = dim + 2
            if value.ndim != expected_ndim:
                raise ValueError(f"Expected a {expected_ndim}D tensor for a {dim}D FNO, got {value.ndim}D")
            if self.channel_last:
                value = self.lift(value).movedim(-1, 1)
            else:
                value = self.lift(value.movedim(1, -1)).movedim(-1, 1)
            for index, (spectral, local) in enumerate(zip(self.spectral_layers, self.local_layers)):
                value = spectral(value) + local(value)
                if index + 1 < len(self.spectral_layers):
                    value = self.activation(value)
            value = self.projection(value.movedim(1, -1))
            return value if self.channel_last else value.movedim(-1, 1)

    return FourierNeuralOperator()


NATIVE_BUILDERS = {
    "pinn": build_pinn,
    "deeponet": build_deeponet,
    "fno": build_fno,
}


__all__ = [
    "MissingTorchDependency",
    "NATIVE_BUILDERS",
    "build_deeponet",
    "build_fno",
    "build_pinn",
]
