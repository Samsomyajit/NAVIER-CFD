from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Mapping

from ..datasets.core import CFDSample


@dataclass(frozen=True)
class DatasetModelDefaults:
    dataset_id: str
    dimension: int
    representation: str
    input_channels: int
    output_channels: int
    coordinate_dim: int
    sensor_count: int
    temporal: bool
    geometry_conditioned: bool
    normalization: str = "channelwise_standard"
    layout: str = "channel_last"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dimension": self.dimension,
            "representation": self.representation,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "coordinate_dim": self.coordinate_dim,
            "sensor_count": self.sensor_count,
            "temporal": self.temporal,
            "geometry_conditioned": self.geometry_conditioned,
            "normalization": self.normalization,
            "layout": self.layout,
        }


DATASET_MODEL_DEFAULTS: dict[str, DatasetModelDefaults] = {
    "pdebench": DatasetModelDefaults("pdebench", 2, "structured", 4, 4, 2, 256, True, False),
    "cfdbench": DatasetModelDefaults("cfdbench", 2, "structured", 3, 3, 2, 256, True, True),
    "realpdebench": DatasetModelDefaults("realpdebench", 2, "structured", 4, 4, 2, 256, True, True),
    "the_well": DatasetModelDefaults("the_well", 3, "structured", 4, 4, 3, 512, True, True),
    "apebench": DatasetModelDefaults("apebench", 2, "structured", 1, 1, 2, 256, True, False),
    "scalarflow": DatasetModelDefaults("scalarflow", 3, "structured", 1, 1, 3, 512, True, False),
    "airfrans": DatasetModelDefaults("airfrans", 2, "point_cloud", 5, 4, 2, 1024, False, True),
    "drivaernetpp": DatasetModelDefaults("drivaernetpp", 3, "point_cloud", 6, 4, 3, 2048, False, True),
    "drivaerml": DatasetModelDefaults("drivaerml", 3, "unstructured", 6, 5, 3, 2048, True, True),
    "shapenet_car": DatasetModelDefaults("shapenet_car", 3, "point_cloud", 3, 4, 3, 2048, False, True),
    "eagle": DatasetModelDefaults("eagle", 3, "unstructured", 4, 4, 3, 1024, True, True),
}


SPECTRAL_MODELS = {
    "fno",
    "pino",
    "f_fno",
    "laplace_no",
}

FIELD_MODELS = {
    "fno",
    "pino",
    "u_fno",
    "f_fno",
    "u_no",
    "lsm",
    "mwt",
    "laplace_no",
    "state_space_no",
    "p3d",
    "tadpole",
}

COORDINATE_MODELS = {"pinn", "nsfnets"}
BRANCH_TRUNK_MODELS = {"deeponet"}

FIELD_COORDINATE_MODELS = {
    "pibert",
    "mionet",
    "fourier_deeponet",
    "nested_fourier_deeponet",
    "fourier_mionet",
    "geo_fno",
    "gino",
    "gnot",
    "galerkin_transformer",
    "factformer",
    "ono",
    "transolver",
    "pinnsformer",
    "pi_mfm",
    "riemannonet",
    "deepmmnet",
    "meshgraphnets",
    "upt",
    "dpot",
    "poseidon",
    "prose_fd",
    "bcat",
    "pdeformer1",
    "aerotransformer",
    "revit",
}


@dataclass(frozen=True)
class ModelBuildPlan:
    model_id: str
    builder_kwargs: Mapping[str, Any]
    input_mode: str
    dataset_id: str | None = None
    dataset_configuration: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "builder_kwargs": dict(self.builder_kwargs),
            "input_mode": self.input_mode,
            "dataset_id": self.dataset_id,
            "dataset_configuration": dict(self.dataset_configuration),
            "notes": list(self.notes),
        }


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("Sample arrays must expose shape")
    return tuple(int(item) for item in shape)


def _resolve_dataset_defaults(dataset_id: str | None) -> DatasetModelDefaults | None:
    if dataset_id is None:
        return None
    key = dataset_id.strip().lower().replace("-", "_")
    aliases = {
        "drivaernet++": "drivaernetpp",
        "drivaernet_plus_plus": "drivaernetpp",
        "shapenetcar": "shapenet_car",
        "real_pde_bench": "realpdebench",
        "cfd_bench": "cfdbench",
        "pde_bench": "pdebench",
    }
    key = aliases.get(key, key)
    try:
        return DATASET_MODEL_DEFAULTS[key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_MODEL_DEFAULTS))
        raise KeyError(f"Unknown dataset configuration {dataset_id!r}; choose one of {available}") from exc


def _sample_properties(
    sample: CFDSample | None,
    defaults: DatasetModelDefaults | None,
    task: Any | None,
) -> dict[str, Any]:
    if sample is not None:
        input_shape = _shape(sample.inputs)
        target_shape = _shape(sample.targets)
        input_channels = input_shape[-1] if len(input_shape) > 1 else 1
        output_channels = target_shape[-1] if len(target_shape) > 1 else 1
        coordinate_dim = None
        if sample.coordinates is not None:
            coordinate_shape = _shape(sample.coordinates)
            coordinate_dim = coordinate_shape[-1] if len(coordinate_shape) > 1 else 1
        dimension = getattr(task, "dimension", None) or coordinate_dim or max(1, len(input_shape) - 1)
        sensor_count = max(1, prod(input_shape[:-1]))
        return {
            "input_shape": input_shape,
            "target_shape": target_shape,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "coordinate_dim": coordinate_dim or dimension,
            "dimension": int(dimension),
            "sensor_count": sensor_count,
        }

    if defaults is None:
        dimension = int(getattr(task, "dimension", 2) or 2)
        return {
            "input_shape": None,
            "target_shape": None,
            "input_channels": 1,
            "output_channels": 1,
            "coordinate_dim": dimension,
            "dimension": dimension,
            "sensor_count": 256,
        }

    return {
        "input_shape": None,
        "target_shape": None,
        "input_channels": defaults.input_channels,
        "output_channels": defaults.output_channels,
        "coordinate_dim": defaults.coordinate_dim,
        "dimension": int(getattr(task, "dimension", None) or defaults.dimension),
        "sensor_count": defaults.sensor_count,
    }


def _dataset_configuration(defaults: DatasetModelDefaults | None, properties: Mapping[str, Any]) -> dict[str, Any]:
    if defaults is None:
        return {
            "representation": "inferred",
            "dimension": properties["dimension"],
            "input_channels": properties["input_channels"],
            "output_channels": properties["output_channels"],
            "coordinate_dim": properties["coordinate_dim"],
            "normalization": "user_defined",
            "layout": "channel_last",
        }
    resolved = defaults.to_dict()
    resolved.update(
        {
            "dimension": properties["dimension"],
            "input_channels": properties["input_channels"],
            "output_channels": properties["output_channels"],
            "coordinate_dim": properties["coordinate_dim"],
            "sensor_count": properties["sensor_count"],
        }
    )
    return resolved


def translate_model_config(
    model_id: str,
    sample: CFDSample | None = None,
    *,
    dataset_id: str | None = None,
    task: Any | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ModelBuildPlan:
    """Resolve model construction from a dataset name and optional canonical sample.

    The dataset supplies representation-aware defaults. A real sample, when given,
    always takes precedence for channel counts, dimensionality, and sensor count.
    User overrides take precedence over both and are recorded in the build plan.
    """

    model_key = model_id.strip().lower()
    defaults = _resolve_dataset_defaults(dataset_id)
    properties = _sample_properties(sample, defaults, task)
    dimension = properties["dimension"]
    input_channels = properties["input_channels"]
    output_channels = properties["output_channels"]
    coordinate_dim = properties["coordinate_dim"]
    representation = defaults.representation if defaults is not None else "inferred"
    geometry = defaults.geometry_conditioned if defaults is not None else False

    notes: list[str] = []
    if sample is None and defaults is not None:
        notes.append("Dataset defaults were used because no canonical sample was supplied; verify field channels against the selected release")
    if representation in {"point_cloud", "unstructured"}:
        notes.append("Variable node counts are padded by the NAVIER-CFD loader and accompanied by a validity mask")

    if model_key == "fno":
        modes = 8 if dimension == 3 else 16
        kwargs: dict[str, Any] = {
            "dimension": dimension,
            "in_channels": input_channels,
            "out_channels": output_channels,
            "modes": tuple(modes for _ in range(dimension)),
            "width": 48 if dimension == 3 else 64,
            "n_layers": 4,
        }
        mode = "field"
    elif model_key == "pibert":
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "coordinate_dim": coordinate_dim,
            "hidden_dim": 160 if dimension == 3 else 128,
            "num_layers": 5 if geometry else 4,
            "num_heads": 8,
            "attention_chunk_size": 512 if representation != "structured" else None,
        }
        mode = "field_with_coordinates"
    elif model_key in COORDINATE_MODELS:
        kwargs = {
            "input_dim": coordinate_dim,
            "coordinate_dim": coordinate_dim,
            "output_dim": output_channels,
            "hidden_dim": 128,
            "hidden_channels": 128,
            "num_layers": 5,
            "depth": 5,
        }
        if model_key == "pinn":
            kwargs.pop("hidden_dim")
            kwargs.pop("num_layers")
        else:
            kwargs.pop("hidden_channels")
            kwargs.pop("depth")
        mode = "coordinates"
        notes.append("PDE, boundary, initial-condition, and closure residuals remain experiment-specific")
    elif model_key in BRANCH_TRUNK_MODELS:
        kwargs = {
            "branch_input_dim": properties["sensor_count"] * input_channels,
            "trunk_input_dim": coordinate_dim,
            "output_dim": output_channels,
            "latent_dim": 128,
            "hidden_channels": 128,
            "depth": 3,
        }
        mode = "branch_trunk"
        notes.append("Branch inputs are flattened; large fields should use explicit sensor subsampling")
    elif model_key in SPECTRAL_MODELS:
        modes = 8 if dimension == 3 else 16
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "dimension": dimension,
            "modes": tuple(modes for _ in range(dimension)),
            "width": 48 if dimension == 3 else 64,
            "num_layers": 4,
        }
        mode = "field"
        if representation != "structured":
            notes.append("Spectral operators require gridding or latent projection for non-structured data")
    elif model_key in FIELD_MODELS:
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "dimension": dimension,
            "width": 48 if dimension == 3 else 64,
            "latent_width": 128,
            "num_layers": 4,
        }
        mode = "field"
        if representation != "structured":
            notes.append("This field operator expects a structured tensor; configure a gridding adapter for point or mesh data")
    elif model_key in FIELD_COORDINATE_MODELS:
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "coordinate_dim": coordinate_dim,
            "dimension": dimension,
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 8,
            "k_neighbors": 12,
        }
        mode = "field_with_coordinates"
    else:
        kwargs = {
            "input_dim": input_channels,
            "output_dim": output_channels,
            "coordinate_dim": coordinate_dim,
            "dimension": dimension,
        }
        mode = "field_with_coordinates" if geometry else "field"
        notes.append("Generic native or external translation; validate the constructor and tensor contract")

    # Remove arguments that are harmless for generic wrappers but invalid for known builders.
    if model_key in {"mionet", "fourier_deeponet", "nested_fourier_deeponet", "fourier_mionet", "deepmmnet"}:
        kwargs.pop("dimension", None)
        kwargs.pop("num_layers", None)
        kwargs.pop("num_heads", None)
        kwargs.pop("k_neighbors", None)
    if model_key in {"geo_fno", "gino", "gnot", "galerkin_transformer", "factformer", "ono", "transolver", "pinnsformer", "pi_mfm", "riemannonet", "meshgraphnets", "upt", "dpot", "poseidon", "prose_fd", "bcat", "pdeformer1", "aerotransformer", "revit"}:
        kwargs.pop("dimension", None)
        if model_key not in {"gino", "meshgraphnets"}:
            kwargs.pop("k_neighbors", None)
    if model_key in {"gino", "meshgraphnets"}:
        kwargs.pop("num_heads", None)

    kwargs.update(dict(overrides or {}))
    dataset_key = defaults.dataset_id if defaults is not None else dataset_id
    return ModelBuildPlan(
        model_id=model_key,
        builder_kwargs=kwargs,
        input_mode=mode,
        dataset_id=dataset_key,
        dataset_configuration=_dataset_configuration(defaults, properties),
        notes=tuple(notes),
    )


__all__ = [
    "BRANCH_TRUNK_MODELS",
    "COORDINATE_MODELS",
    "DATASET_MODEL_DEFAULTS",
    "DatasetModelDefaults",
    "FIELD_COORDINATE_MODELS",
    "FIELD_MODELS",
    "ModelBuildPlan",
    "SPECTRAL_MODELS",
    "translate_model_config",
]
