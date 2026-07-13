from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .catalogs import Catalog
from .datasets import AdaptedDataset, AdapterRegistry, HuggingFaceDatasetManager, make_dataloaders
from .models import ModelHub, translate_model_config
from .specs import TaskSpec
from .training import CFDTrainer, TrainerConfig, TrainingResult


@dataclass
class ExperimentResult:
    model_id: str
    dataset_id: str
    training: TrainingResult
    metrics: Mapping[str, float]
    build_plan: Mapping[str, Any]
    manifest_path: str | None = None


@dataclass
class Experiment:
    """High-level model–dataset experiment orchestration.

    Resolved adapters, model dimensions, split seed, and training settings remain
    explicit and are saved in a reproducible manifest.
    """

    dataset_id: str
    model_id: str
    task: TaskSpec
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    batch_size: int = 8
    split_seed: int = 0
    adapter_options: Mapping[str, Any] = field(default_factory=dict)
    model_overrides: Mapping[str, Any] = field(default_factory=dict)
    output_dir: str | None = None

    def prepare(self, raw_dataset: Any) -> tuple[Any, dict[str, Any], Any]:
        adapter = AdapterRegistry().adapter(self.dataset_id, **dict(self.adapter_options))
        dataset = AdaptedDataset(raw_dataset, adapter)
        if len(dataset) < 1:
            raise ValueError("The dataset is empty")
        sample = dataset[0]
        plan = translate_model_config(
            self.model_id,
            sample,
            task=self.task,
            overrides=self.model_overrides,
        )
        model = ModelHub().load(
            self.model_id,
            task=self.task,
            **dict(plan.builder_kwargs),
        )
        loaders = make_dataloaders(dataset, batch_size=self.batch_size, seed=self.split_seed)
        return model, loaders, plan

    def run(self, raw_dataset: Any, *, velocity_metrics: bool = False) -> ExperimentResult:
        model, loaders, plan = self.prepare(raw_dataset)
        config = self.trainer_config
        if self.output_dir and not config.checkpoint_dir:
            config = TrainerConfig(
                **{
                    **asdict(config),
                    "checkpoint_dir": str(Path(self.output_dir) / "checkpoints"),
                }
            )
        trainer = CFDTrainer(model, model_id=self.model_id, config=config)
        training = trainer.fit(loaders["train"], loaders.get("validation"))
        metrics = trainer.evaluate(loaders["test"], velocity=velocity_metrics)
        manifest_path = self._write_manifest(plan, training, metrics) if self.output_dir else None
        return ExperimentResult(
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            training=training,
            metrics=metrics,
            build_plan={
                "model_id": plan.model_id,
                "builder_kwargs": dict(plan.builder_kwargs),
                "input_mode": plan.input_mode,
                "notes": list(plan.notes),
            },
            manifest_path=manifest_path,
        )

    def load_huggingface(
        self,
        *,
        split: str | None = None,
        config: str | None = None,
        streaming: bool = False,
        revision: str | None = None,
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        dataset_spec = Catalog.load_builtin().dataset(self.dataset_id)
        return HuggingFaceDatasetManager(token=token).load(
            dataset_spec,
            split=split,
            config=config,
            streaming=streaming,
            revision=revision,
            **kwargs,
        )

    def _write_manifest(self, plan: Any, training: TrainingResult, metrics: Mapping[str, float]) -> str:
        import json

        directory = Path(self.output_dir or ".")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "experiment-manifest.json"
        payload = {
            "schema": "navier-cfd.experiment/v1",
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "task": self.task.to_dict(),
            "adapter_options": dict(self.adapter_options),
            "model_plan": {
                "builder_kwargs": dict(plan.builder_kwargs),
                "input_mode": plan.input_mode,
                "notes": list(plan.notes),
            },
            "trainer": asdict(self.trainer_config),
            "split_seed": self.split_seed,
            "training": {
                "best_epoch": training.best_epoch,
                "best_validation_loss": training.best_validation_loss,
                "checkpoint": training.checkpoint,
            },
            "metrics": dict(metrics),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)


__all__ = ["Experiment", "ExperimentResult"]
