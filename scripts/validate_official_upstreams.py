from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from navier_cfd import OfficialDatasetManager, load_cfd_dataset


PUBLIC_PROBE_DATASETS = (
    "airfrans",
    "drivaernetpp",
    "drivaerml",
    "scalarflow",
    "shapenet_car",
    "eagle",
)


def _finite_sample(sample: Any) -> dict[str, Any]:
    inputs = np.asarray(sample.inputs)
    targets = np.asarray(sample.targets)
    if inputs.size == 0 or targets.size == 0:
        raise RuntimeError("Official sample contains empty inputs or targets")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise RuntimeError("Official sample contains non-finite values")
    return {
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "coordinate_shape": None
        if sample.coordinates is None
        else list(np.asarray(sample.coordinates).shape),
        "provider": sample.metadata.get("provider"),
        "source_file": sample.metadata.get("source_file"),
        "input_fields": list(sample.metadata.get("input_fields", ())),
        "target_fields": list(sample.metadata.get("target_fields", ())),
    }


def validate_probes(output: Path) -> dict[str, Any]:
    manager = OfficialDatasetManager()
    report: dict[str, Any] = {"grade": "official_endpoint_verified", "datasets": {}}
    failures = []
    for dataset_id in PUBLIC_PROBE_DATASETS:
        probe = manager.probe(dataset_id, timeout=30.0, max_entries=15)
        report["datasets"][dataset_id] = probe.to_dict()
        if not probe.reachable:
            failures.append(f"{dataset_id}: {probe.error_category}: {probe.message}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise RuntimeError("Official upstream probes failed:\n" + "\n".join(failures))
    return report


def validate_drivaerml(output_dir: Path) -> dict[str, Any]:
    manager = OfficialDatasetManager()
    repo = manager.source("drivaerml").repository
    assert repo is not None
    # A full boundary file is about 660 MB. This publisher-hosted CFD slice is small enough
    # for pull-request CI while retaining real flow-field variables in VTP format.
    requested = "run_1/slices/xNormal_m15000.vtp"
    revision = "main"
    resolved_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{requested}"
    request = Request(
        resolved_url,
        method="HEAD",
        headers={"User-Agent": "navier-cfd-upstream-validation/0.8"},
    )
    with urlopen(request, timeout=60.0) as response:
        final_url = response.geturl()
        content_length = response.headers.get("Content-Length")
        linked_size = response.headers.get("X-Linked-Size")
        size_text = linked_size or content_length
        size = None if size_text is None else int(size_text)
        status = getattr(response, "status", 200)
    limit = 50 * 1024**2
    if status >= 400:
        raise RuntimeError(f"Official DrivAerML smoke file returned HTTP {status}")
    if size is not None and size > limit:
        raise RuntimeError(f"Official DrivAerML smoke file is too large for CI: {size} bytes")
    result = manager.download(
        "drivaerml",
        output_dir,
        artifacts=[requested],
        revision=revision,
    )
    dataset = load_cfd_dataset(
        "drivaerml",
        local_path=result.files[0],
        split="all",
        max_samples=1,
        max_windows=1,
    )
    sample = dataset[0]
    report = {
        "grade": "real_official_file_loaded",
        "dataset": "drivaerml",
        "upstream_file": requested,
        "upstream_url": resolved_url,
        "resolved_url": final_url,
        "upstream_size": size,
        "download": result.to_dict(),
        "sample": _finite_sample(sample),
    }
    (output_dir / "drivaerml-live-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def validate_apebench(output_dir: Path) -> dict[str, Any]:
    dataset = load_cfd_dataset(
        "apebench",
        configuration="Advection",
        scenario_group="difficulty",
        split="train",
        scenario_kwargs={
            "num_points": 32,
            "num_train_samples": 2,
            "train_temporal_horizon": 4,
        },
        n_steps_input=2,
        n_steps_output=1,
        max_samples=1,
        max_windows=1,
    )
    report = {
        "grade": "official_procedural_sample_generated",
        "dataset": "apebench",
        "access_plan": dataset.access_plan,
        "sample": _finite_sample(dataset[0]),
    }
    (output_dir / "apebench-live-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def validate_drivaernetpp_metadata(output_dir: Path) -> dict[str, Any]:
    result = OfficialDatasetManager().download(
        "drivaernetpp",
        output_dir,
        artifacts=["train_split", "validation_split", "test_split"],
        max_bytes=10 * 1024**2,
    )
    counts = {}
    for filename in result.files:
        path = Path(filename)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"Official split file is empty: {path}")
        counts[path.name] = len(lines)
    report = {
        "grade": "official_file_downloaded",
        "dataset": "drivaernetpp",
        "download": result.to_dict(),
        "split_counts": counts,
        "cfd_note": "Full CFD files require a selected Harvard Dataverse file ID or Globus transfer.",
    }
    (output_dir / "drivaernetpp-live-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("probes", "drivaerml", "apebench", "drivaernetpp"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("upstream-validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "probes":
        report = validate_probes(args.output_dir / "official-probes.json")
    elif args.mode == "drivaerml":
        report = validate_drivaerml(args.output_dir)
    elif args.mode == "apebench":
        report = validate_apebench(args.output_dir)
    else:
        report = validate_drivaernetpp_metadata(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
