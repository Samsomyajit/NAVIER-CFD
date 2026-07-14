from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalogs import Catalog
from .datasets import AdaptedDataset, AdapterRegistry, HuggingFaceDatasetManager, load_cfd_dataset, make_dataloaders
from .metrics import MetricContext
from .models import load_model
from .specs import TaskSpec
from .training import CFDTrainer, TrainerConfig, TrainingResult


@dataclass
class ExperimentResult:
    model_id: str
    dataset_id: str
    training: TrainingResult
    metrics: Mapping[str, Any]
    build_plan: Mapping[str, Any]
    manifest_path: str | None = None


@dataclass
class Experiment:
    """High-level dataset-aware model training and evaluation orchestration."""

    dataset_id: str
    model_id: str
    task: TaskSpec
    dataset_configuration: str | None = None
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    batch_size: int = 8
    split_seed: int = 0
    adapter_options: Mapping[str, Any] = field(default_factory=dict)
    model_overrides: Mapping[str, Any] = field(default_factory=dict)
    metric_suites: tuple[str, ...] = ("data_standard",)
    metric_context: MetricContext = field(default_factory=MetricContext)
    output_dir: str | None = None

    def prepare(self, raw_dataset: Any) -> tuple[Any, dict[str, Any], Any]:
        if self.dataset_id == "the_well" and hasattr(raw_dataset, "adapter"):
            dataset = raw_dataset
        else:
            adapter = AdapterRegistry().adapter(self.dataset_id, **dict(self.adapter_options))
            dataset = AdaptedDataset(raw_dataset, adapter)
        if len(dataset) < 1:
            raise ValueError("The dataset is empty")
        sample = dataset[0]
        model, plan = load_model(
            self.model_id,
            dataset=self.dataset_id,
            sample=sample,
            task=self.task,
            overrides=self.model_overrides,
            return_plan=True,
        )
        loaders = make_dataloaders(dataset, batch_size=self.batch_size, seed=self.split_seed)
        return model, loaders, plan

    def run(
        self,
        raw_dataset: Any,
        *,
        velocity_metrics: bool = False,
        metric_suites: str | Sequence[str] | None = None,
        metric_context: MetricContext | None = None,
    ) -> ExperimentResult:
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
        selected_suites = metric_suites or self.metric_suites
        metrics = trainer.evaluate(
            loaders["test"],
            velocity=velocity_metrics,
            metric_suites=selected_suites,
            metric_context=metric_context or self.metric_context,
            include_metric_records=True,
        )
        manifest_path = self._write_manifest(plan, training, metrics) if self.output_dir else None
        return ExperimentResult(
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            training=training,
            metrics=metrics,
            build_plan=plan.to_dict(),
            manifest_path=manifest_path,
        )

    def load_dataset(
        self,
        *,
        split: str = "train",
        streaming: bool = False,
        local_path: str | None = None,
        token: str | None = None,
        adapt: bool = True,
        **kwargs: Any,
    ) -> Any:
        return load_cfd_dataset(
            self.dataset_id,
            configuration=self.dataset_configuration,
            split=split,
            streaming=streaming,
            local_path=local_path,
            token=token,
            adapt=adapt,
            **kwargs,
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

    def _write_manifest(self, plan: Any, training: TrainingResult, metrics: Mapping[str, Any]) -> str:
        import json

        directory = Path(self.output_dir or ".")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "experiment-manifest.json"
        payload = {
            "schema": "navier-cfd.experiment/v3",
            "dataset_id": self.dataset_id,
            "dataset_configuration": self.dataset_configuration,
            "dataset_provider": Catalog.load_builtin().dataset(self.dataset_id).provider,
            "model_id": self.model_id,
            "task": self.task.to_dict(),
            "adapter_options": dict(self.adapter_options),
            "model_plan": plan.to_dict(),
            "trainer": asdict(self.trainer_config),
            "split_seed": self.split_seed,
            "metric_suites": list(self.metric_suites),
            "metric_context": asdict(self.metric_context),
            "training": {
                "best_epoch": training.best_epoch,
                "best_validation_loss": training.best_validation_loss,
                "checkpoint": training.checkpoint,
            },
            "metrics": dict(metrics),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)


__all__ = ["Experiment", "ExperimentResult"]
