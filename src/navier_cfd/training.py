from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .benchmarks.metrics import compute_metric_bundle
from .checkpoints import CheckpointManager
from .datasets.core import CFDBatch


@dataclass
class TrainerConfig:
    epochs: int = 100
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    scheduler: str | None = "cosine"
    gradient_clip: float | None = 1.0
    mixed_precision: bool = True
    device: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0
    early_stopping_patience: int | None = None


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    checkpoint: str | None = None


class CFDTrainer:
    """Common supervised trainer for native and adapted neural CFD models."""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        config: TrainerConfig | None = None,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        forward_fn: Callable[[Any, CFDBatch], Any] | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Training requires PyTorch; install `navier-cfd[models]`") from exc
        self.torch = torch
        self.nn = nn
        self.model = model
        self.model_id = model_id
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_fn = loss_fn or self._build_loss(self.config.loss)
        self.forward_fn = forward_fn
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.checkpoints = CheckpointManager()

    def _build_loss(self, name: str) -> Any:
        key = name.lower()
        if key == "mse":
            return self.nn.MSELoss()
        if key in {"mae", "l1"}:
            return self.nn.L1Loss()
        if key in {"huber", "smooth_l1"}:
            return self.nn.SmoothL1Loss()
        raise ValueError(f"Unknown loss {name!r}")

    def _build_optimizer(self) -> Any:
        key = self.config.optimizer.lower()
        kwargs = {"lr": self.config.learning_rate, "weight_decay": self.config.weight_decay}
        if key == "adam":
            return self.torch.optim.Adam(self.model.parameters(), **kwargs)
        if key == "adamw":
            return self.torch.optim.AdamW(self.model.parameters(), **kwargs)
        if key == "sgd":
            return self.torch.optim.SGD(self.model.parameters(), momentum=0.9, **kwargs)
        if key == "lbfgs":
            return self.torch.optim.LBFGS(self.model.parameters(), lr=self.config.learning_rate)
        raise ValueError(f"Unknown optimizer {self.config.optimizer!r}")

    def _build_scheduler(self) -> Any | None:
        if not self.config.scheduler:
            return None
        key = self.config.scheduler.lower()
        if key == "cosine":
            return self.torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config.epochs),
            )
        if key == "plateau":
            return self.torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min")
        raise ValueError(f"Unknown scheduler {self.config.scheduler!r}")

    def _forward(self, batch: CFDBatch) -> Any:
        if self.forward_fn is not None:
            return self.forward_fn(self.model, batch)
        if self.model_id == "pibert":
            return self.model(batch.inputs, coordinates=batch.coordinates, mask=batch.mask)
        if self.model_id == "pinn":
            values = batch.coordinates if batch.coordinates is not None else batch.inputs
            return self.model(values)
        if self.model_id == "deeponet":
            if batch.coordinates is None:
                raise ValueError("DeepONet requires coordinates in the canonical batch")
            branch = batch.inputs.reshape(batch.inputs.shape[0], -1)
            trunk = batch.coordinates.reshape(
                batch.coordinates.shape[0],
                -1,
                batch.coordinates.shape[-1],
            )
            return self.model(branch, trunk)
        return self.model(batch.inputs)

    @staticmethod
    def _align(prediction: Any, target: Any) -> tuple[Any, Any]:
        if prediction.shape == target.shape:
            return prediction, target
        if prediction.numel() == target.numel():
            return prediction.reshape_as(target), target
        raise ValueError(f"Prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}")

    def _masked_loss(self, prediction: Any, target: Any, mask: Any | None) -> Any:
        prediction, target = self._align(prediction, target)
        if mask is None:
            return self.loss_fn(prediction, target)
        expanded = mask
        while expanded.ndim < prediction.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(prediction)
        return self.loss_fn(prediction[expanded], target[expanded])

    def _run_epoch(self, loader: Any, *, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        count = 0
        amp_enabled = self.scaler.is_enabled()
        for batch in loader:
            batch = batch.to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with self.torch.set_grad_enabled(training):
                with self.torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    prediction = self._forward(batch)
                    loss = self._masked_loss(prediction, batch.targets, batch.mask)
            if training:
                if isinstance(self.optimizer, self.torch.optim.LBFGS):
                    def closure() -> Any:
                        self.optimizer.zero_grad(set_to_none=True)
                        pred = self._forward(batch)
                        value = self._masked_loss(pred, batch.targets, batch.mask)
                        value.backward()
                        return value

                    loss = self.optimizer.step(closure)
                else:
                    self.scaler.scale(loss).backward()
                    if self.config.gradient_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            total += float(loss.detach().cpu())
            count += 1
        return total / max(count, 1)

    def fit(self, train_loader: Any, validation_loader: Any | None = None) -> TrainingResult:
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_epoch = 0
        best_checkpoint: str | None = None
        stale = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = (
                self._run_epoch(validation_loader, training=False)
                if validation_loader is not None
                else train_loss
            )
            if self.scheduler is not None:
                if isinstance(self.scheduler, self.torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(validation_loss)
                else:
                    self.scheduler.step()
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(row)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
                if self.config.checkpoint_dir:
                    path = Path(self.config.checkpoint_dir) / "best"
                    self.checkpoints.save(
                        path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        config=asdict(self.config),
                        metrics=row,
                        metadata={"model_id": self.model_id},
                        epoch=epoch,
                    )
                    best_checkpoint = str(path)
            else:
                stale += 1

            if (
                self.config.checkpoint_dir
                and self.config.checkpoint_every > 0
                and epoch % self.config.checkpoint_every == 0
            ):
                self.checkpoints.save(
                    Path(self.config.checkpoint_dir) / f"epoch-{epoch:04d}",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=asdict(self.config),
                    metrics=row,
                    metadata={"model_id": self.model_id},
                    epoch=epoch,
                )
            if self.config.early_stopping_patience is not None and stale >= self.config.early_stopping_patience:
                break

        return TrainingResult(history, best_epoch, best_loss, best_checkpoint)

    def predict(self, loader: Any) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        targets = []
        with self.torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                prediction = self._forward(batch)
                prediction, target = self._align(prediction, batch.targets)
                predictions.append(prediction.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
        return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)

    def evaluate(
        self,
        loader: Any,
        *,
        velocity: bool = False,
        spacing: tuple[float, ...] | None = None,
    ) -> Mapping[str, float]:
        prediction, target = self.predict(loader)
        return compute_metric_bundle(prediction, target, velocity=velocity, spacing=spacing)


__all__ = ["CFDTrainer", "TrainerConfig", "TrainingResult"]
