from __future__ import annotations

import math
from typing import Any, Sequence


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PIBERT requires PyTorch; install `navier-cfd[models]`") from exc
    return torch, nn, functional


class FourierCoordinateEmbedding:
    """Factory wrapper returning a torch module without importing torch at package import."""

    def __new__(cls, coordinate_dim: int, hidden_dim: int, num_frequencies: int = 16) -> Any:
        torch, nn, _ = _require_torch()
        if coordinate_dim < 1 or num_frequencies < 1:
            raise ValueError("coordinate_dim and num_frequencies must be positive")

        class _Embedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                frequencies = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
                self.register_buffer("frequencies", frequencies, persistent=False)
                self.projection = nn.Linear(coordinate_dim * num_frequencies * 2, hidden_dim)

            def forward(self, coordinates: Any) -> Any:
                phase = 2.0 * math.pi * coordinates.unsqueeze(-1) * self.frequencies
                features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
                return self.projection(features.flatten(-2))

        return _Embedding()


class MultiscaleWaveletEmbedding:
    """Sequence-local multiscale detail features inspired by Haar decompositions."""

    def __new__(cls, hidden_dim: int, scales: Sequence[int] = (1, 2, 4)) -> Any:
        torch, nn, functional = _require_torch()
        scales = tuple(int(scale) for scale in scales)
        if not scales or any(scale < 1 for scale in scales):
            raise ValueError("wavelet scales must be positive")

        class _Wavelet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(hidden_dim * (len(scales) + 1), hidden_dim)

            def forward(self, values: Any) -> Any:
                channel_first = values.transpose(1, 2)
                details = [values]
                for scale in scales:
                    kernel = 2 * scale + 1
                    smooth = functional.avg_pool1d(
                        channel_first,
                        kernel_size=kernel,
                        stride=1,
                        padding=scale,
                        count_include_pad=False,
                    ).transpose(1, 2)
                    details.append(values - smooth)
                return self.projection(torch.cat(details, dim=-1))

        return _Wavelet()


class PhysicsBiasedAttention:
    def __new__(
        cls,
        hidden_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        distance_bias: float = 1.0,
        residual_bias: float = 0.25,
    ) -> Any:
        torch, nn, _ = _require_torch()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = hidden_dim // num_heads

        class _Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.scale = head_dim**-0.5
                self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
                self.output = nn.Linear(hidden_dim, hidden_dim)
                self.dropout = nn.Dropout(dropout)
                self.distance_strength = nn.Parameter(torch.tensor(float(distance_bias)))
                self.residual_strength = nn.Parameter(torch.tensor(float(residual_bias)))

            def _physics_bias(self, coordinates: Any | None, residuals: Any | None, length: int) -> Any | None:
                bias = None
                if coordinates is not None:
                    if coordinates.ndim == 2:
                        coordinates = coordinates.unsqueeze(0)
                    distance = torch.cdist(coordinates.float(), coordinates.float())
                    scale = distance.detach().mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
                    bias = -self.distance_strength.abs() * distance / scale
                if residuals is not None:
                    if residuals.ndim == 2:
                        magnitude = residuals.abs()
                    else:
                        magnitude = residuals.reshape(residuals.shape[0], length, -1).norm(dim=-1)
                    magnitude = magnitude / magnitude.detach().mean(dim=-1, keepdim=True).clamp_min(1e-6)
                    focus = 0.5 * (magnitude.unsqueeze(-1) + magnitude.unsqueeze(-2))
                    residual_term = self.residual_strength * focus
                    bias = residual_term if bias is None else bias + residual_term
                return bias

            def forward(
                self,
                values: Any,
                *,
                coordinates: Any | None = None,
                residuals: Any | None = None,
                mask: Any | None = None,
            ) -> Any:
                batch, length, _ = values.shape
                qkv = self.qkv(values).reshape(batch, length, 3, self.num_heads, self.head_dim)
                query, key, value = qkv.unbind(dim=2)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
                bias = self._physics_bias(coordinates, residuals, length)
                if bias is not None:
                    if bias.shape[0] == 1 and batch > 1:
                        bias = bias.expand(batch, -1, -1)
                    logits = logits + bias.unsqueeze(1)
                if mask is not None:
                    if mask.ndim > 2:
                        mask = mask.reshape(mask.shape[0], -1)
                    key_mask = ~mask.bool().unsqueeze(1).unsqueeze(2)
                    logits = logits.masked_fill(key_mask, torch.finfo(logits.dtype).min)
                attention = torch.softmax(logits, dim=-1)
                attention = self.dropout(attention)
                output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, -1)
                output = self.output(output)
                if mask is not None:
                    output = output * mask.unsqueeze(-1).to(output.dtype)
                return output

        return _Attention()


class PIBERTBlock:
    def __new__(
        cls,
        hidden_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> Any:
        _, nn, _ = _require_torch()
        inner = max(hidden_dim, int(hidden_dim * mlp_ratio))
        attention = PhysicsBiasedAttention(hidden_dim, num_heads, dropout=dropout)

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.attention = attention
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, inner),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(inner, hidden_dim),
                    nn.Dropout(dropout),
                )

            def forward(self, values: Any, **kwargs: Any) -> Any:
                values = values + self.attention(self.norm1(values), **kwargs)
                return values + self.mlp(self.norm2(values))

        return _Block()


class PIBERT:
    """Functional Fourier-wavelet physics-biased transformer for CFD fields.

    This implementation follows the public PIBERT design description while
    replacing the placeholder linear layer in the original repository with
    executable spectral embeddings, multiscale details, physics-biased attention,
    transformer blocks, and an output head.
    """

    def __new__(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        coordinate_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        num_frequencies: int = 16,
        wavelet_scales: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        use_fourier: bool = True,
        use_wavelet: bool = True,
    ) -> Any:
        torch, nn, _ = _require_torch()
        if min(input_dim, output_dim, coordinate_dim, hidden_dim, num_layers, num_heads) < 1:
            raise ValueError("PIBERT dimensions and layer counts must be positive")

        class _PIBERT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.coordinate_dim = coordinate_dim
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                self.fourier = (
                    FourierCoordinateEmbedding(coordinate_dim, hidden_dim, num_frequencies)
                    if use_fourier
                    else None
                )
                self.wavelet = MultiscaleWaveletEmbedding(hidden_dim, wavelet_scales) if use_wavelet else None
                self.blocks = nn.ModuleList(
                    [
                        PIBERTBlock(
                            hidden_dim,
                            num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.norm = nn.LayerNorm(hidden_dim)
                self.output_projection = nn.Linear(hidden_dim, output_dim)

            def _flatten(self, values: Any) -> tuple[Any, tuple[int, ...]]:
                if values.ndim < 3:
                    raise ValueError("PIBERT input must have shape [B,L,C] or [B,*spatial,C]")
                spatial_shape = tuple(values.shape[1:-1])
                return values.reshape(values.shape[0], -1, values.shape[-1]), spatial_shape

            def _default_coordinates(self, batch: int, spatial_shape: tuple[int, ...], device: Any, dtype: Any) -> Any:
                axes = [torch.linspace(0.0, 1.0, steps=size, device=device, dtype=dtype) for size in spatial_shape]
                mesh = torch.meshgrid(*axes, indexing="ij")
                coords = torch.stack(mesh, dim=-1).reshape(-1, len(spatial_shape))
                if coords.shape[-1] < self.coordinate_dim:
                    padding = torch.zeros(
                        coords.shape[0],
                        self.coordinate_dim - coords.shape[-1],
                        device=device,
                        dtype=dtype,
                    )
                    coords = torch.cat((coords, padding), dim=-1)
                elif coords.shape[-1] > self.coordinate_dim:
                    coords = coords[:, : self.coordinate_dim]
                return coords.unsqueeze(0).expand(batch, -1, -1)

            def _coordinates(self, coordinates: Any | None, values: Any, spatial_shape: tuple[int, ...]) -> Any:
                batch = values.shape[0]
                if coordinates is None:
                    return self._default_coordinates(batch, spatial_shape, values.device, values.dtype)
                if coordinates.ndim == len(spatial_shape) + 1:
                    coordinates = coordinates.reshape(-1, coordinates.shape[-1]).unsqueeze(0)
                elif coordinates.ndim == len(spatial_shape) + 2:
                    coordinates = coordinates.reshape(batch, -1, coordinates.shape[-1])
                elif coordinates.ndim == 2:
                    coordinates = coordinates.unsqueeze(0)
                if coordinates.shape[0] == 1 and batch > 1:
                    coordinates = coordinates.expand(batch, -1, -1)
                if coordinates.shape[-1] != self.coordinate_dim:
                    if coordinates.shape[-1] < self.coordinate_dim:
                        padding = torch.zeros(
                            *coordinates.shape[:-1],
                            self.coordinate_dim - coordinates.shape[-1],
                            device=coordinates.device,
                            dtype=coordinates.dtype,
                        )
                        coordinates = torch.cat((coordinates, padding), dim=-1)
                    else:
                        coordinates = coordinates[..., : self.coordinate_dim]
                return coordinates

            def forward(
                self,
                values: Any,
                coordinates: Any | None = None,
                pde_residuals: Any | None = None,
                mask: Any | None = None,
                *,
                coords: Any | None = None,
            ) -> Any:
                if coords is not None and coordinates is None:
                    coordinates = coords
                sequence, spatial_shape = self._flatten(values)
                coordinates = self._coordinates(coordinates, sequence, spatial_shape)
                hidden = self.input_projection(sequence)
                if self.fourier is not None:
                    hidden = hidden + self.fourier(coordinates)
                if self.wavelet is not None:
                    hidden = hidden + self.wavelet(hidden)
                flat_mask = mask
                if flat_mask is not None:
                    flat_mask = flat_mask.reshape(flat_mask.shape[0], -1)
                residuals = pde_residuals
                if residuals is not None:
                    residuals = residuals.reshape(residuals.shape[0], -1, residuals.shape[-1])
                for block in self.blocks:
                    hidden = block(
                        hidden,
                        coordinates=coordinates,
                        residuals=residuals,
                        mask=flat_mask,
                    )
                output = self.output_projection(self.norm(hidden))
                return output.reshape(values.shape[0], *spatial_shape, self.output_dim)

            def predict(self, values: Any, coords: Any | None = None) -> Any:
                return self(values, coords=coords)

        return _PIBERT()


def build_pibert(
    *,
    task: Any | None = None,
    input_dim: int,
    output_dim: int,
    coordinate_dim: int | None = None,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    num_frequencies: int = 16,
    wavelet_scales: Sequence[int] = (1, 2, 4),
    dropout: float = 0.0,
    mlp_ratio: float = 4.0,
    use_fourier: bool = True,
    use_wavelet: bool = True,
    **_: Any,
) -> Any:
    return PIBERT(
        input_dim=input_dim,
        output_dim=output_dim,
        coordinate_dim=coordinate_dim or getattr(task, "dimension", None) or 2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_frequencies=num_frequencies,
        wavelet_scales=wavelet_scales,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        use_fourier=use_fourier,
        use_wavelet=use_wavelet,
    )


__all__ = ["PIBERT", "build_pibert"]
