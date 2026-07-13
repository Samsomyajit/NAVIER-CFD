from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


class CheckpointError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


class CheckpointManager:
    """Portable directory checkpoints with weights, optimizer state, and manifest."""

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise CheckpointError("Checkpoint support requires PyTorch") from exc
        return torch

    def save(
        self,
        path: str | Path,
        *,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        config: Any | None = None,
        metrics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        epoch: int | None = None,
    ) -> Path:
        torch = self._torch()
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory / "weights.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), directory / "scheduler.pt")
        manifest = {
            "schema": "navier-cfd.checkpoint/v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "config": _jsonable(config),
            "metrics": _jsonable(metrics or {}),
            "metadata": _jsonable(metadata or {}),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return directory

    def load(
        self,
        path: str | Path,
        *,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        map_location: Any = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        torch = self._torch()
        directory = Path(path)
        weights = directory / "weights.pt"
        manifest_path = directory / "manifest.json"
        if not weights.exists() or not manifest_path.exists():
            raise CheckpointError(f"Invalid checkpoint directory: {directory}")
        state = torch.load(weights, map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=strict)
        if optimizer is not None and (directory / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(directory / "optimizer.pt", map_location=map_location, weights_only=True)
            )
        if scheduler is not None and (directory / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(directory / "scheduler.pt", map_location=map_location, weights_only=True)
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = ["CheckpointError", "CheckpointManager"]
