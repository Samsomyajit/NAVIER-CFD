from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .audit import audit_figure_spec
from .specs import FigureManifest, FigureSpec


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Figure rendering requires the optional dependency: pip install 'navier-cfd[autoresearch]'"
        ) from exc
    return plt


def _save(fig, output: Path, spec: FigureSpec) -> tuple[str, ...]:
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    paths: list[str] = []
    for suffix in spec.output_formats:
        path = stem.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = spec.dpi
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    return tuple(paths)


def render_truth_prediction_error(
    truth: np.ndarray,
    prediction: np.ndarray,
    spec: FigureSpec,
    output: str | Path,
    *,
    coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    mask: np.ndarray | None = None,
    figure_id: str = "field-comparison",
    manifest_metadata: Mapping[str, object] | None = None,
) -> FigureManifest:
    if spec.figure_type != "truth_prediction_error":
        raise ValueError("render_truth_prediction_error requires a truth_prediction_error FigureSpec")
    audit = audit_figure_spec(spec)
    if not audit.valid:
        messages = "; ".join(issue.message for issue in audit.issues if issue.severity == "error")
        raise ValueError(f"Figure specification failed audit: {messages}")

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Truth and prediction must be matching 2D arrays")
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
    shown_truth = np.where(valid, truth, np.nan)
    shown_prediction = np.where(valid, prediction, np.nan)
    raw_error = prediction - truth
    if spec.error_definition == "absolute":
        error = np.abs(raw_error)
    elif spec.error_definition == "squared":
        error = raw_error**2
    elif spec.error_definition == "relative":
        scale = np.maximum(np.abs(truth), np.finfo(float).eps)
        error = raw_error / scale
    else:
        error = raw_error
    shown_error = np.where(valid, error, np.nan)

    if spec.color_limits is None:
        combined = np.concatenate((shown_truth[valid], shown_prediction[valid]))
        vmin, vmax = float(np.min(combined)), float(np.max(combined))
    else:
        vmin, vmax = spec.color_limits
    if spec.error_limits is None:
        emin, emax = float(np.nanmin(shown_error)), float(np.nanmax(shown_error))
        if spec.error_definition == "signed":
            bound = max(abs(emin), abs(emax))
            emin, emax = -bound, bound
    else:
        emin, emax = spec.error_limits

    plt = _pyplot()
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "savefig.transparent": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
        extent = None
        if coordinates is not None:
            x, y = coordinates
            extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
        panels = (
            (shown_truth, "Ground truth", vmin, vmax, "viridis"),
            (shown_prediction, "Prediction", vmin, vmax, "viridis"),
            (shown_error, "Error", emin, emax, "coolwarm" if spec.error_definition == "signed" else "magma"),
        )
        for axis, (values, title, low, high, cmap) in zip(axes, panels):
            image = axis.imshow(
                values,
                origin="lower",
                aspect="equal",
                extent=extent,
                interpolation="none",
                vmin=low,
                vmax=high,
                cmap=cmap,
            )
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            if spec.units:
                colorbar.set_label(spec.units if title != "Error" else f"error ({spec.units})")
        if spec.title:
            fig.suptitle(spec.title)
        outputs = _save(fig, Path(output), spec)
        plt.close(fig)

    manifest = FigureManifest(
        figure_id=figure_id,
        spec=spec,
        outputs=outputs,
        metadata={
            "audit": audit.to_dict(),
            "shape": list(truth.shape),
            **dict(manifest_metadata or {}),
        },
    )
    manifest.save(Path(output).with_suffix(".manifest.json"))
    return manifest


def render_profile(
    coordinate: np.ndarray,
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    spec: FigureSpec,
    output: str | Path,
    *,
    figure_id: str = "profile-comparison",
) -> FigureManifest:
    if spec.figure_type != "profile":
        raise ValueError("render_profile requires a profile FigureSpec")
    audit = audit_figure_spec(spec)
    if not audit.valid:
        raise ValueError("Figure specification failed audit")
    coordinate = np.asarray(coordinate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if coordinate.shape != truth.shape:
        raise ValueError("coordinate and truth must have the same shape")
    plt = _pyplot()
    with plt.rc_context({"font.family": "serif", "font.size": 9}):
        fig, axis = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
        axis.plot(truth, coordinate, linewidth=1.8, label="Ground truth")
        for label, values in predictions.items():
            values = np.asarray(values, dtype=float)
            if values.shape != truth.shape:
                raise ValueError(f"Prediction {label!r} shape does not match truth")
            axis.plot(values, coordinate, linewidth=1.3, label=label)
        axis.set_xlabel(f"{spec.fields[0]} ({spec.units})" if spec.units else spec.fields[0])
        axis.set_ylabel("Coordinate")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        if spec.title:
            axis.set_title(spec.title)
        outputs = _save(fig, Path(output), spec)
        plt.close(fig)
    manifest = FigureManifest(
        figure_id=figure_id,
        spec=spec,
        outputs=outputs,
        metadata={"audit": audit.to_dict(), "models": list(predictions)},
    )
    manifest.save(Path(output).with_suffix(".manifest.json"))
    return manifest
