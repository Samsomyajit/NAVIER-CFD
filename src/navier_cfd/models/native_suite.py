from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .native import MissingTorchDependency, build_fno


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingTorchDependency(
            "Native NAVIER-CFD models require PyTorch. Install `navier-cfd[models]`."
        ) from exc
    return torch, nn, functional


def _activation(nn: Any, name: str) -> Any:
    table = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported activation {name!r}")
    return table[key]


def _flatten_field(values: Any) -> tuple[Any, tuple[int, ...]]:
    if values.ndim < 3:
        raise ValueError("Field models expect [batch, spatial..., channels]")
    spatial = tuple(int(item) for item in values.shape[1:-1])
    return values.reshape(values.shape[0], -1, values.shape[-1]), spatial


def _restore_field(values: Any, spatial: Sequence[int]) -> Any:
    return values.reshape(values.shape[0], *spatial, values.shape[-1])


def _coordinates_for(values: Any, coordinates: Any | None, coordinate_dim: int) -> Any:
    torch, _, _ = _require_torch()
    sequence, spatial = _flatten_field(values)
    if coordinates is not None:
        coordinates = coordinates.reshape(values.shape[0], -1, coordinates.shape[-1])
        if coordinates.shape[-1] < coordinate_dim:
            padding = torch.zeros(
                coordinates.shape[0], coordinates.shape[1], coordinate_dim - coordinates.shape[-1],
                dtype=coordinates.dtype, device=coordinates.device,
            )
            coordinates = torch.cat((coordinates, padding), dim=-1)
        return coordinates[..., :coordinate_dim]
    axes = [torch.linspace(-1.0, 1.0, size, device=values.device, dtype=values.dtype) for size in spatial]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(mesh, dim=-1).reshape(1, sequence.shape[1], len(spatial))
    if grid.shape[-1] < coordinate_dim:
        padding = torch.zeros(
            grid.shape[0], grid.shape[1], coordinate_dim - grid.shape[-1],
            dtype=grid.dtype, device=grid.device,
        )
        grid = torch.cat((grid, padding), dim=-1)
    return grid[..., :coordinate_dim].expand(values.shape[0], -1, -1)


@dataclass(frozen=True)
class NativeModelFamily:
    id: str
    builder: Callable[..., Any]
    input_mode: str = "field"


class NativeFieldMLP:
    def __new__(
        cls,
        input_channels: int,
        output_channels: int,
        hidden_dim: int = 64,
        depth: int = 3,
        activation: str = "gelu",
        **_: Any,
    ) -> Any:
        _, nn, _ = _require_torch()
        layers: list[Any] = []
        width = input_channels
        for _index in range(max(1, depth)):
            layers.extend((nn.Linear(width, hidden_dim), _activation(nn, activation)()))
            width = hidden_dim
        layers.append(nn.Linear(width, output_channels))
        return nn.Sequential(*layers)


class NativeResidualFieldModel:
    def __new__(
        cls,
        input_channels: int,
        output_channels: int,
        hidden_dim: int = 64,
        depth: int = 3,
        activation: str = "gelu",
        **_: Any,
    ) -> Any:
        torch, nn, _ = _require_torch()

        class ResidualModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Linear(input_channels, hidden_dim)
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            _activation(nn, activation)(),
                            nn.Linear(hidden_dim, hidden_dim),
                        )
                        for _index in range(max(1, depth))
                    ]
                )
                self.output = nn.Linear(hidden_dim, output_channels)

            def forward(self, values: Any) -> Any:
                state = _activation(nn, activation)()(self.input(values))
                scale = torch.tensor(float(len(self.blocks)), dtype=state.dtype, device=state.device)
                for block in self.blocks:
                    state = state + block(state) / scale
                return self.output(state)

        return ResidualModel()


class NativeCoordinateOperator:
    def __new__(
        cls,
        input_channels: int,
        output_channels: int,
        coordinate_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        activation: str = "gelu",
        **_: Any,
    ) -> Any:
        _, nn, _ = _require_torch()

        class CoordinateOperator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.coordinate_dim = coordinate_dim
                self.network = NativeResidualFieldModel(
                    input_channels=input_channels + coordinate_dim,
                    output_channels=output_channels,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    activation=activation,
                )

            def forward(self, values: Any, coordinates: Any | None = None) -> Any:
                sequence, spatial = _flatten_field(values)
                grid = _coordinates_for(values, coordinates, self.coordinate_dim)
                output = self.network(nn.functional.pad(sequence, (0, 0)))
                del output
                prediction = self.network(
                    nn.functional.pad(
                        nn.functional.pad(sequence, (0, self.coordinate_dim)),
                        (0, 0),
                    )
                )
                prediction = self.network(nn.functional.pad(sequence, (0, self.coordinate_dim)))
                prediction = self.network(
                    nn.functional.pad(sequence, (0, self.coordinate_dim))
                    + nn.functional.pad(grid, (input_channels, 0))
                )
                return _restore_field(prediction, spatial)

        return CoordinateOperator()


class NativeSpectralReference:
    def __new__(
        cls,
        input_channels: int,
        output_channels: int,
        dimension: int = 2,
        width: int = 32,
        modes: int | Sequence[int] = 12,
        depth: int = 4,
        **kwargs: Any,
    ) -> Any:
        return build_fno(
            input_channels=input_channels,
            output_channels=output_channels,
            dimension=dimension,
            width=width,
            modes=modes,
            depth=depth,
            **kwargs,
        )


NATIVE_REFERENCE_FAMILIES: Mapping[str, NativeModelFamily] = {
    "field_mlp": NativeModelFamily("field_mlp", NativeFieldMLP),
    "residual_field": NativeModelFamily("residual_field", NativeResidualFieldModel),
    "coordinate_operator": NativeModelFamily("coordinate_operator", NativeCoordinateOperator),
    "spectral": NativeModelFamily("spectral", NativeSpectralReference),
}


__all__ = [
    "NATIVE_REFERENCE_FAMILIES",
    "NativeCoordinateOperator",
    "NativeFieldMLP",
    "NativeModelFamily",
    "NativeResidualFieldModel",
    "NativeSpectralReference",
]
