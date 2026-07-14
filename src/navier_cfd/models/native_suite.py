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
        padding = torch.zeros(1, grid.shape[1], coordinate_dim - grid.shape[-1], device=values.device, dtype=values.dtype)
        grid = torch.cat((grid, padding), dim=-1)
    return grid[..., :coordinate_dim].expand(values.shape[0], -1, -1)


class PointTransformerOperator:
    """Coordinate-aware transformer used by several native reference families."""

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
        residual_output: bool = False,
        **_: Any,
    ) -> Any:
        torch, nn, _ = _require_torch()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        act = _activation(nn, activation)

        class _Operator(nn.Module):
            navier_input_mode = "field_with_coordinates"

            def __init__(self) -> None:
                super().__init__()
                self.coordinate_dim = coordinate_dim
                self.input_projection = nn.Linear(input_dim + coordinate_dim, hidden_dim)
                block = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(block, num_layers=num_layers)
                self.output = nn.Sequential(nn.LayerNorm(hidden_dim), act(), nn.Linear(hidden_dim, output_dim))
                self.residual_output = residual_output and input_dim == output_dim

            def forward(self, values: Any, coordinates: Any | None = None, mask: Any | None = None, **__: Any) -> Any:
                sequence, spatial = _flatten_field(values)
                coords = _coordinates_for(values, coordinates, self.coordinate_dim)
                hidden = self.input_projection(torch.cat((sequence, coords), dim=-1))
                padding_mask = None
                if mask is not None:
                    padding_mask = ~mask.reshape(mask.shape[0], -1).bool()
                hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
                output = self.output(hidden)
                if self.residual_output:
                    output = output + sequence
                return _restore_field(output, spatial)

        return _Operator()


class GraphOperator:
    """Dense local-message reference graph operator for point and mesh data."""

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int = 2,
        hidden_dim: int = 96,
        num_layers: int = 4,
        k_neighbors: int = 12,
        **_: Any,
    ) -> Any:
        torch, nn, _ = _require_torch()

        class _Graph(nn.Module):
            navier_input_mode = "field_with_coordinates"

            def __init__(self) -> None:
                super().__init__()
                self.coordinate_dim = coordinate_dim
                self.k_neighbors = k_neighbors
                self.node = nn.Linear(input_dim + coordinate_dim, hidden_dim)
                self.edge_mlps = nn.ModuleList(
                    [nn.Sequential(nn.Linear(hidden_dim * 2 + coordinate_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)]
                )
                self.node_mlps = nn.ModuleList(
                    [nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)]
                )
                self.output = nn.Linear(hidden_dim, output_dim)

            def forward(self, values: Any, coordinates: Any | None = None, mask: Any | None = None, **__: Any) -> Any:
                sequence, spatial = _flatten_field(values)
                coords = _coordinates_for(values, coordinates, self.coordinate_dim)
                hidden = self.node(torch.cat((sequence, coords), dim=-1))
                count = hidden.shape[1]
                distances = torch.cdist(coords, coords)
                k = min(max(1, self.k_neighbors), count)
                neighbors = distances.topk(k=k, largest=False).indices
                batch_index = torch.arange(hidden.shape[0], device=hidden.device)[:, None, None]
                for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
                    neighbor_hidden = hidden[batch_index, neighbors]
                    neighbor_coords = coords[batch_index, neighbors]
                    center = hidden[:, :, None, :].expand_as(neighbor_hidden)
                    delta = neighbor_coords - coords[:, :, None, :]
                    messages = edge_mlp(torch.cat((center, neighbor_hidden, delta), dim=-1)).mean(dim=2)
                    hidden = hidden + node_mlp(torch.cat((hidden, messages), dim=-1))
                    if mask is not None:
                        hidden = hidden * mask.reshape(mask.shape[0], -1, 1)
                return _restore_field(self.output(hidden), spatial)

        return _Graph()


class ConvOperator:
    """Dimension-generic residual convolutional operator."""

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        dimension: int = 2,
        width: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        multiscale: bool = False,
        **_: Any,
    ) -> Any:
        _, nn, _ = _require_torch()
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimension]
        padding = kernel_size // 2

        class _Conv(nn.Module):
            navier_input_mode = "field"

            def __init__(self) -> None:
                super().__init__()
                self.lift = conv(input_dim, width, 1)
                self.blocks = nn.ModuleList()
                for layer in range(num_layers):
                    dilation = 2 ** (layer % 3) if multiscale else 1
                    block_padding = padding * dilation
                    self.blocks.append(
                        nn.Sequential(
                            conv(width, width, kernel_size, padding=block_padding, dilation=dilation),
                            nn.GELU(),
                            conv(width, width, 1),
                        )
                    )
                self.output = conv(width, output_dim, 1)

            def forward(self, values: Any, **__: Any) -> Any:
                hidden = values.movedim(-1, 1)
                hidden = self.lift(hidden)
                for block in self.blocks:
                    hidden = hidden + block(hidden)
                return self.output(hidden).movedim(1, -1)

        return _Conv()


class LatentOperator:
    """Autoencoder-style latent operator for U-NO, LSM and related variants."""

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        dimension: int = 2,
        width: int = 64,
        latent_width: int = 128,
        num_layers: int = 3,
        **_: Any,
    ) -> Any:
        _, nn, functional = _require_torch()
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimension]
        pool = {1: nn.avg_pool1d, 2: nn.avg_pool2d, 3: nn.avg_pool3d}[dimension]
        mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[dimension]

        class _Latent(nn.Module):
            navier_input_mode = "field"

            def __init__(self) -> None:
                super().__init__()
                self.lift = conv(input_dim, width, 1)
                self.encoder = nn.Sequential(conv(width, latent_width, 3, padding=1), nn.GELU())
                self.blocks = nn.ModuleList(
                    [nn.Sequential(conv(latent_width, latent_width, 3, padding=1), nn.GELU(), conv(latent_width, latent_width, 1)) for _ in range(num_layers)]
                )
                self.output = conv(latent_width, output_dim, 1)

            def forward(self, values: Any, **__: Any) -> Any:
                original = tuple(values.shape[1:-1])
                hidden = self.encoder(self.lift(values.movedim(-1, 1)))
                if min(hidden.shape[2:]) >= 2:
                    hidden = pool(hidden, kernel_size=2, stride=2)
                for block in self.blocks:
                    hidden = hidden + block(hidden)
                hidden = functional.interpolate(hidden, size=original, mode=mode, align_corners=False)
                return self.output(hidden).movedim(1, -1)

        return _Latent()


class PhysicsMLP:
    """Coordinate-network reference model with optional field conditioning."""

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 5,
        activation: str = "tanh",
        condition_on_fields: bool = False,
        **_: Any,
    ) -> Any:
        torch, nn, _ = _require_torch()
        act = _activation(nn, activation)
        coord_dim = coordinate_dim or input_dim
        network_input = coord_dim + (input_dim if condition_on_fields else 0)
        layers: list[Any] = [nn.Linear(network_input, hidden_dim), act()]
        for _ in range(max(1, num_layers - 1)):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), act()))
        layers.append(nn.Linear(hidden_dim, output_dim))

        class _Physics(nn.Module):
            navier_input_mode = "coordinates" if not condition_on_fields else "field_with_coordinates"

            def __init__(self) -> None:
                super().__init__()
                self.coordinate_dim = coord_dim
                self.network = nn.Sequential(*layers)

            def forward(self, values: Any, coordinates: Any | None = None, **__: Any) -> Any:
                if condition_on_fields:
                    sequence, spatial = _flatten_field(values)
                    coords = _coordinates_for(values, coordinates, self.coordinate_dim)
                    return _restore_field(self.network(torch.cat((coords, sequence), dim=-1)), spatial)
                coords = coordinates if coordinates is not None else values
                return self.network(coords)

            def residual(self, coordinates: Any, operator: Callable[[Any, Any], Any]) -> Any:
                coordinates = coordinates.requires_grad_(True)
                prediction = self.network(coordinates)
                return operator(prediction, coordinates)

        return _Physics()


class MultiInputOperator:
    """MIONet-style multiplicative fusion with optional Fourier coordinate features."""

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int = 2,
        hidden_dim: int = 128,
        num_inputs: int = 2,
        fourier: bool = False,
        **_: Any,
    ) -> Any:
        torch, nn, _ = _require_torch()

        class _Multi(nn.Module):
            navier_input_mode = "field_with_coordinates"

            def __init__(self) -> None:
                super().__init__()
                self.coordinate_dim = coordinate_dim
                self.num_inputs = max(1, num_inputs)
                self.encoders = nn.ModuleList(
                    [nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(self.num_inputs)]
                )
                coord_width = coordinate_dim * (5 if fourier else 1)
                self.coordinate = nn.Sequential(nn.Linear(coord_width, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
                self.output = nn.Linear(hidden_dim, output_dim)
                frequencies = torch.arange(1, 3, dtype=torch.float32)
                self.register_buffer("frequencies", frequencies, persistent=False)
                self.fourier = fourier

            def _coord_features(self, coords: Any) -> Any:
                if not self.fourier:
                    return coords
                angles = coords[..., None] * self.frequencies * torch.pi
                return torch.cat((coords, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)

            def forward(self, values: Any, coordinates: Any | None = None, **__: Any) -> Any:
                tensors = list(values) if isinstance(values, (tuple, list)) else [values] * self.num_inputs
                sequence, spatial = _flatten_field(tensors[0])
                fused = None
                for encoder, tensor in zip(self.encoders, tensors):
                    current = encoder(_flatten_field(tensor)[0])
                    fused = current if fused is None else fused * current
                coords = _coordinates_for(tensors[0], coordinates, self.coordinate_dim)
                fused = fused * self.coordinate(self._coord_features(coords))
                return _restore_field(self.output(fused), spatial)

        return _Multi()


@dataclass(frozen=True)
class NativeFamily:
    builder: Callable[..., Any]
    defaults: Mapping[str, Any]


def _point_builder(**defaults: Any) -> Callable[..., Any]:
    return lambda **kwargs: PointTransformerOperator(**{**defaults, **kwargs})


def _graph_builder(**defaults: Any) -> Callable[..., Any]:
    return lambda **kwargs: GraphOperator(**{**defaults, **kwargs})


def _conv_builder(**defaults: Any) -> Callable[..., Any]:
    return lambda **kwargs: ConvOperator(**{**defaults, **kwargs})


def _latent_builder(**defaults: Any) -> Callable[..., Any]:
    return lambda **kwargs: LatentOperator(**{**defaults, **kwargs})


def _physics_builder(**defaults: Any) -> Callable[..., Any]:
    return lambda **kwargs: PhysicsMLP(**{**defaults, **kwargs})


def _multi_builder(**defaults: Any) -> Callable[..., Any]:
    return lambda **kwargs: MultiInputOperator(**{**defaults, **kwargs})


def _fno_reference(**defaults: Any) -> Callable[..., Any]:
    def build(**kwargs: Any) -> Any:
        merged = {**defaults, **kwargs}
        if "input_dim" in merged:
            merged.setdefault("in_channels", merged.pop("input_dim"))
        if "output_dim" in merged:
            merged.setdefault("out_channels", merged.pop("output_dim"))
        if "num_layers" in merged:
            merged.setdefault("n_layers", merged.pop("num_layers"))
        return build_fno(**merged)
    return build


NATIVE_REFERENCE_FAMILIES: dict[str, NativeFamily] = {
    # Operator learning
    "mionet": NativeFamily(_multi_builder(num_inputs=2), {}),
    "fourier_deeponet": NativeFamily(_multi_builder(num_inputs=1, fourier=True), {}),
    "nested_fourier_deeponet": NativeFamily(_multi_builder(num_inputs=2, fourier=True), {}),
    "fourier_mionet": NativeFamily(_multi_builder(num_inputs=3, fourier=True), {}),
    "pino": NativeFamily(_fno_reference(), {}),
    "geo_fno": NativeFamily(_point_builder(num_layers=3), {}),
    "gino": NativeFamily(_graph_builder(num_layers=4), {}),
    "u_fno": NativeFamily(_latent_builder(num_layers=3), {}),
    "f_fno": NativeFamily(_fno_reference(width=48), {}),
    "u_no": NativeFamily(_latent_builder(num_layers=4), {}),
    "lsm": NativeFamily(_latent_builder(latent_width=160), {}),
    "gnot": NativeFamily(_point_builder(num_layers=4), {}),
    "galerkin_transformer": NativeFamily(_point_builder(num_layers=4), {}),
    "mwt": NativeFamily(_conv_builder(multiscale=True, num_layers=5), {}),
    "factformer": NativeFamily(_point_builder(num_layers=3), {}),
    "ono": NativeFamily(_point_builder(num_layers=3), {}),
    "transolver": NativeFamily(_point_builder(num_layers=6), {}),
    "laplace_no": NativeFamily(_fno_reference(width=48), {}),
    "state_space_no": NativeFamily(_conv_builder(multiscale=True, num_layers=6), {}),
    # Physics-informed ML
    "nsfnets": NativeFamily(_physics_builder(condition_on_fields=False), {}),
    "pinnsformer": NativeFamily(_point_builder(num_layers=4), {}),
    "pi_mfm": NativeFamily(_point_builder(num_layers=5), {}),
    "riemannonet": NativeFamily(_physics_builder(condition_on_fields=True), {}),
    "deepmmnet": NativeFamily(_multi_builder(num_inputs=2), {}),
    # Deep learning / geometry / foundation references
    "meshgraphnets": NativeFamily(_graph_builder(num_layers=5), {}),
    "upt": NativeFamily(_point_builder(num_layers=6), {}),
    "dpot": NativeFamily(_point_builder(num_layers=6), {}),
    "poseidon": NativeFamily(_point_builder(num_layers=6), {}),
    "prose_fd": NativeFamily(_point_builder(num_layers=5), {}),
    "bcat": NativeFamily(_point_builder(num_layers=5), {}),
    "pdeformer1": NativeFamily(_point_builder(num_layers=5), {}),
    "p3d": NativeFamily(_conv_builder(multiscale=True, num_layers=6), {}),
    "aerotransformer": NativeFamily(_point_builder(num_layers=6), {}),
    "tadpole": NativeFamily(_latent_builder(num_layers=5), {}),
    "revit": NativeFamily(_point_builder(num_layers=4), {}),
}


def build_native_reference(model_id: str, **kwargs: Any) -> Any:
    try:
        family = NATIVE_REFERENCE_FAMILIES[model_id]
    except KeyError as exc:
        raise KeyError(f"No native reference builder registered for {model_id}") from exc
    model = family.builder(**{**family.defaults, **kwargs})
    model.navier_reference_model_id = model_id
    model.navier_reference_notice = (
        "NAVIER-CFD native reference implementation; validate against the official paper and repository before claiming reproduction."
    )
    return model


def reference_builder(model_id: str) -> Callable[..., Any]:
    def build(*, task: Any | None = None, spec: Any | None = None, **kwargs: Any) -> Any:
        kwargs.pop("task", None)
        kwargs.pop("spec", None)
        return build_native_reference(model_id, **kwargs)
    build.__name__ = f"build_{model_id}"
    return build


REFERENCE_BUILDERS = {model_id: reference_builder(model_id) for model_id in NATIVE_REFERENCE_FAMILIES}


__all__ = [
    "NATIVE_REFERENCE_FAMILIES",
    "REFERENCE_BUILDERS",
    "build_native_reference",
]
