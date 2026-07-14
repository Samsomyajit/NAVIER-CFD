from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core import CFDSample


@dataclass(frozen=True)
class ScientificDatasetAccessPlan:
    provider: str
    dataset_id: str
    configuration: str
    split: str
    repo_id: str
    revision: str | None
    resolved_revision: str | None
    files: tuple[str, ...]
    subset_mode: bool
    official_split: bool
    auth_source: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ListCFDDataset:
    def __init__(
        self,
        samples: Sequence[CFDSample],
        *,
        access_plan: Mapping[str, Any] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.access_plan = dict(access_plan or {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CFDSample:
        return self.samples[index]


def split_members(
    count: int,
    split: str,
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[int]:
    if count < 1:
        return []
    normalized = {"val": "validation", "valid": "validation"}.get(split, split)
    if normalized not in {"train", "validation", "test", "all"}:
        raise ValueError("split must be train, validation/valid/val, test, or all")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    if normalized == "all":
        return indices
    train_stop = max(1, int(round(fractions[0] * count)))
    valid_stop = min(count, train_stop + max(1, int(round(fractions[1] * count))))
    partitions = {
        "train": indices[:train_stop],
        "validation": indices[train_stop:valid_stop],
        "test": indices[valid_stop:],
    }
    if not partitions[normalized]:
        partitions[normalized] = indices[-1:]
    return partitions[normalized]


def flatten_time_channels(array: np.ndarray) -> np.ndarray:
    """Convert ``[time, spatial..., channels]`` to ``[spatial..., time*channels]``."""

    if array.ndim < 3:
        raise ValueError("Expected [time, spatial..., channels]")
    moved = np.moveaxis(array, 0, -2)
    return moved.reshape(*moved.shape[:-2], moved.shape[-2] * moved.shape[-1])


def ensure_channel_last_time_series(
    array: np.ndarray,
    *,
    time_axis: int = 0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Normalize one trajectory to ``[time, spatial..., channels]``."""

    value = np.asarray(array)
    if value.ndim < 2:
        raise ValueError("A trajectory must have at least time and one spatial axis")
    value = np.moveaxis(value, time_axis, 0)
    if channel_axis is not None:
        adjusted = channel_axis
        if adjusted < 0:
            adjusted += array.ndim
        if adjusted == time_axis:
            raise ValueError("time_axis and channel_axis cannot be identical")
        if adjusted > time_axis:
            adjusted -= 1
        value = np.moveaxis(value, adjusted + 1, -1)
        return value
    if value.ndim >= 4 and value.shape[-1] <= 16:
        return value
    return value[..., None]


def coordinate_grid(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    vectors = [np.asarray(vector) for vector in vectors if vector is not None]
    if not vectors:
        return None
    if all(vector.ndim == 1 for vector in vectors):
        return np.stack(np.meshgrid(*vectors, indexing="ij"), axis=-1).astype(np.float32)
    if len(vectors) == 1 and vectors[0].ndim >= 2 and vectors[0].shape[-1] in (1, 2, 3):
        return vectors[0].astype(np.float32, copy=False)
    if len(vectors) in (2, 3) and all(vector.shape == vectors[0].shape for vector in vectors):
        return np.stack(vectors, axis=-1).astype(np.float32)
    return None


def first_matching_path(
    paths: Iterable[str],
    *,
    suffixes: tuple[str, ...],
    contains: str | None = None,
) -> str | None:
    candidates = []
    for path in paths:
        lower = path.lower()
        if not lower.endswith(suffixes):
            continue
        if contains and contains.lower() not in lower:
            continue
        candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def resolve_cache_dir(cache_dir: str | Path | None, provider: str) -> Path:
    if cache_dir is None:
        root = Path.home() / ".cache" / "navier-cfd" / provider
    else:
        root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


__all__ = [
    "ListCFDDataset",
    "ScientificDatasetAccessPlan",
    "coordinate_grid",
    "ensure_channel_last_time_series",
    "first_matching_path",
    "flatten_time_channels",
    "resolve_cache_dir",
    "split_members",
]
