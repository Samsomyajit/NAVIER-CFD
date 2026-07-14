from __future__ import annotations

import importlib
import re
from typing import Any, Mapping

import numpy as np

from ..core import CFDSample, DatasetAdapterError
from .common import (
    ListCFDDataset,
    ScientificDatasetAccessPlan,
    coordinate_grid,
    flatten_time_channels,
    split_members,
)


class MissingAPEBenchDependency(RuntimeError):
    """Raised when procedural APEBench data are requested without apebench installed."""


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _load_apebench() -> Any:
    try:
        return importlib.import_module("apebench")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingAPEBenchDependency(
            "APEBench support requires `pip install navier-cfd[apebench]`."
        ) from exc


def _resolve_scenario(apebench: Any, configuration: str, scenario_group: str) -> type[Any]:
    scenarios = getattr(apebench, "scenarios", None)
    namespace = getattr(scenarios, scenario_group, None)
    if namespace is None:
        available = (
            sorted(name for name in dir(scenarios) if not name.startswith("_"))
            if scenarios is not None
            else []
        )
        raise ValueError(
            f"Unknown APEBench scenario group {scenario_group!r}; available groups: {available}"
        )

    requested = _normalise_name(configuration)
    matches: list[type[Any]] = []
    for name in dir(namespace):
        if name.startswith("_"):
            continue
        candidate = getattr(namespace, name)
        if isinstance(candidate, type) and _normalise_name(name) == requested:
            matches.append(candidate)
    if not matches:
        available = sorted(
            name
            for name in dir(namespace)
            if not name.startswith("_") and isinstance(getattr(namespace, name), type)
        )
        raise ValueError(
            f"Unknown APEBench scenario {configuration!r} in group {scenario_group!r}; "
            f"available scenarios include {available[:20]}"
        )
    return matches[0]


def _trajectory_samples(
    data: np.ndarray,
    *,
    split: str,
    scenario_name: str,
    configuration: str,
    scenario_group: str,
    n_steps_input: int,
    n_steps_output: int,
    time_stride: int,
    window_stride: int,
    max_windows: int | None,
    trajectory_indices: list[int],
    access_plan: Mapping[str, Any],
) -> list[CFDSample]:
    if data.ndim < 4:
        raise DatasetAdapterError(
            "APEBench data must have shape [samples, time, channels, spatial...]; "
            f"received {data.shape}"
        )
    if n_steps_input < 1 or n_steps_output < 1:
        raise ValueError("n_steps_input and n_steps_output must be positive")

    samples: list[CFDSample] = []
    horizon = (n_steps_input + n_steps_output - 1) * time_stride + 1
    for trajectory_index in trajectory_indices:
        raw = np.asarray(data[trajectory_index])
        if raw.ndim < 3:
            raise DatasetAdapterError(
                f"APEBench trajectory {trajectory_index} has invalid shape {raw.shape}"
            )
        trajectory = np.moveaxis(raw, 1, -1)
        time_count = trajectory.shape[0]
        spatial_shape = trajectory.shape[1:-1]
        axes = [
            np.linspace(0.0, 1.0, size, dtype=np.float32)
            for size in spatial_shape
        ]
        coordinates = coordinate_grid(axes)
        starts = range(
            0,
            max(0, time_count - horizon + 1),
            max(1, window_stride),
        )
        for start in starts:
            indices = start + np.arange(n_steps_input + n_steps_output) * time_stride
            selected = trajectory[indices]
            inputs = flatten_time_channels(selected[:n_steps_input])
            targets = flatten_time_channels(selected[n_steps_input:])
            samples.append(
                CFDSample(
                    inputs=inputs.astype(np.float32, copy=False),
                    targets=targets.astype(np.float32, copy=False),
                    coordinates=coordinates,
                    metadata={
                        "dataset_id": "apebench",
                        "provider": "apebench_procedural",
                        "configuration": configuration,
                        "scenario_group": scenario_group,
                        "scenario_name": scenario_name,
                        "source_split": split,
                        "trajectory_index": trajectory_index,
                        "window_start": start,
                        "input_steps": n_steps_input,
                        "output_steps": n_steps_output,
                        "time_stride": time_stride,
                        "representation": "structured",
                        "layout": "spatial_channel_last_time_flattened",
                        "access_plan": dict(access_plan),
                    },
                )
            )
            if max_windows is not None and len(samples) >= max_windows:
                return samples
    if not samples:
        raise DatasetAdapterError(
            "No APEBench windows remain; reduce the temporal horizon or increase generated data"
        )
    return samples


class APEBenchDatasetManager:
    """Generate canonical CFD windows through the official APEBench Python API."""

    def probe(
        self,
        configuration: str | None = None,
        *,
        scenario_group: str = "difficulty",
    ) -> dict[str, Any]:
        try:
            apebench = _load_apebench()
        except MissingAPEBenchDependency:
            return {
                "provider": "apebench_procedural",
                "dependency_available": False,
                "install_spec": "navier-cfd[apebench]",
                "configuration": configuration,
                "scenario_group": scenario_group,
                "download_required": False,
            }
        result: dict[str, Any] = {
            "provider": "apebench_procedural",
            "dependency_available": True,
            "configuration": configuration,
            "scenario_group": scenario_group,
            "download_required": False,
        }
        if configuration:
            scenario_class = _resolve_scenario(apebench, configuration, scenario_group)
            result["scenario_class"] = scenario_class.__name__
        return result

    def load(
        self,
        configuration: str,
        *,
        split: str = "train",
        scenario_group: str = "difficulty",
        scenario_kwargs: Mapping[str, Any] | None = None,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        time_stride: int = 1,
        window_stride: int = 1,
        max_samples: int | None = 16,
        max_windows: int | None = 128,
        seed: int = 42,
    ) -> ListCFDDataset:
        apebench = _load_apebench()
        scenario_class = _resolve_scenario(apebench, configuration, scenario_group)
        kwargs = dict(scenario_kwargs or {})
        if max_samples is not None:
            kwargs.setdefault("num_train_samples", max_samples)
            kwargs.setdefault("num_test_samples", max_samples)
        required_horizon = (n_steps_input + n_steps_output - 1) * time_stride + 1
        kwargs.setdefault(
            "train_temporal_horizon",
            max(required_horizon, n_steps_input + n_steps_output),
        )
        kwargs.setdefault(
            "test_temporal_horizon",
            max(required_horizon, n_steps_input + n_steps_output),
        )
        scenario = scenario_class(**kwargs)

        normalized_split = {"val": "validation", "valid": "validation"}.get(split, split)
        if normalized_split == "test":
            data = np.asarray(scenario.get_test_data())
            trajectory_indices = list(range(len(data)))
            official_split = True
        elif normalized_split in {"train", "validation"}:
            data = np.asarray(scenario.get_train_data())
            if normalized_split == "validation":
                trajectory_indices = split_members(len(data), "validation", seed=seed)
                official_split = False
            else:
                trajectory_indices = list(range(len(data)))
                official_split = True
        elif normalized_split == "all":
            train_data = np.asarray(scenario.get_train_data())
            test_data = np.asarray(scenario.get_test_data())
            data = np.concatenate([train_data, test_data], axis=0)
            trajectory_indices = list(range(len(data)))
            official_split = False
        else:
            raise ValueError("split must be train, validation/valid/val, test, or all")

        if max_samples is not None:
            trajectory_indices = trajectory_indices[:max_samples]
        scenario_name = (
            str(scenario.get_scenario_name())
            if hasattr(scenario, "get_scenario_name")
            else scenario_class.__name__
        )
        plan = ScientificDatasetAccessPlan(
            provider="apebench_procedural",
            dataset_id="apebench",
            configuration=configuration,
            split=normalized_split,
            repo_id="pypi:apebench",
            revision=None,
            resolved_revision=getattr(apebench, "__version__", None),
            files=(),
            subset_mode=max_samples is not None or max_windows is not None,
            official_split=official_split,
            auth_source="not_applicable",
            notes=(
                "Data are generated procedurally through the official APEBench scenario API.",
                "Validation is a deterministic subset of the generated training trajectories.",
            ),
        )
        samples = _trajectory_samples(
            data,
            split=normalized_split,
            scenario_name=scenario_name,
            configuration=configuration,
            scenario_group=scenario_group,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            time_stride=time_stride,
            window_stride=window_stride,
            max_windows=max_windows,
            trajectory_indices=trajectory_indices,
            access_plan=plan.to_dict(),
        )
        return ListCFDDataset(samples, access_plan=plan.to_dict())


__all__ = [
    "APEBenchDatasetManager",
    "MissingAPEBenchDependency",
]
