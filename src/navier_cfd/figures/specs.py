from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


FIGURE_TYPES = {
    "truth_prediction_error",
    "profile",
    "rollout",
    "spectrum",
    "parity",
    "ablation",
    "interface_error",
}
OUTPUT_FORMATS = {"pdf", "svg", "png"}
ERROR_DEFINITIONS = {"signed", "absolute", "squared", "relative"}


@dataclass(frozen=True)
class FigureSpec:
    """Reproducible specification for a research-grade CFD figure."""

    figure_type: str
    fields: tuple[str, ...]
    title: str | None = None
    cases: tuple[str, ...] = ()
    times: tuple[float, ...] = ()
    units: str | None = None
    shared_color_limits: bool = True
    color_limits: tuple[float, float] | None = None
    error_limits: tuple[float, float] | None = None
    error_definition: str = "absolute"
    mask: str | None = None
    output_formats: tuple[str, ...] = ("pdf", "svg", "png")
    style: str = "nature_clean"
    dpi: int = 400
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.figure_type not in FIGURE_TYPES:
            raise ValueError(f"Unknown figure_type {self.figure_type!r}; choose from {sorted(FIGURE_TYPES)}")
        if not self.fields:
            raise ValueError("At least one field is required")
        unsupported = set(self.output_formats) - OUTPUT_FORMATS
        if unsupported:
            raise ValueError(f"Unsupported output formats: {sorted(unsupported)}")
        if self.error_definition not in ERROR_DEFINITIONS:
            raise ValueError(f"Unsupported error definition: {self.error_definition}")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")
        for name, limits in (("color_limits", self.color_limits), ("error_limits", self.error_limits)):
            if limits is not None and limits[0] >= limits[1]:
                raise ValueError(f"{name} must be ordered low < high")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureSpec":
        return cls(
            figure_type=str(data["figure_type"]),
            fields=tuple(data.get("fields", ())),
            title=data.get("title"),
            cases=tuple(data.get("cases", ())),
            times=tuple(float(value) for value in data.get("times", ())),
            units=data.get("units"),
            shared_color_limits=bool(data.get("shared_color_limits", True)),
            color_limits=None if data.get("color_limits") is None else tuple(data["color_limits"]),
            error_limits=None if data.get("error_limits") is None else tuple(data["error_limits"]),
            error_definition=str(data.get("error_definition", "absolute")),
            mask=data.get("mask"),
            output_formats=tuple(data.get("output_formats", ("pdf", "svg", "png"))),
            style=str(data.get("style", "nature_clean")),
            dpi=int(data.get("dpi", 400)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class FigureManifest:
    """Traceability sidecar for a rendered scientific figure."""

    figure_id: str
    spec: FigureSpec
    source_run: str | None = None
    source_commit: str | None = None
    dataset_hash: str | None = None
    checkpoint_hash: str | None = None
    normalization: str | None = None
    renderer_version: str = "navier-figurelab-1.1.0"
    outputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "spec": self.spec.to_dict(),
            "source_run": self.source_run,
            "source_commit": self.source_commit,
            "dataset_hash": self.dataset_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "normalization": self.normalization,
            "renderer_version": self.renderer_version,
            "outputs": list(self.outputs),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        import json

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination


def publication_field_comparison(
    field: str,
    *,
    units: str,
    cases: Sequence[str] = (),
    times: Sequence[float] = (),
    color_limits: tuple[float, float] | None = None,
    mask: str | None = None,
) -> FigureSpec:
    return FigureSpec(
        figure_type="truth_prediction_error",
        fields=(field,),
        cases=tuple(cases),
        times=tuple(times),
        units=units,
        shared_color_limits=True,
        color_limits=color_limits,
        error_definition="absolute",
        mask=mask,
    )
