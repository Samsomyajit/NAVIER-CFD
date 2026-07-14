from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalogs import Catalog
from .datasets import (
    AdaptedDataset,
    AdapterRegistry,
    CFDSample,
    HuggingFaceDatasetManager,
    load_cfd_dataset,
    make_dataloaders,
    make_split_dataloaders,
)
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

    def _adapt_dataset(self, raw_dataset: Any) -> Any:
        if hasattr(raw_dataset, "adapter"):
            return raw_dataset
        if hasattr(raw_dataset, "__len__") and hasattr(raw_dataset, "__getitem__"):
            if len(raw_dataset) > 0 and isinstance(raw_dataset[0], CFDSample):
                return raw_dataset
        adapter = AdapterRegistry().adapter(self.dataset_id, **dict(self.adapter_options))
        return AdaptedDataset(raw_dataset, adapter)

    @staticmethod
    def _is_split_mapping(raw_dataset: Any) -> bool:
        if not isinstance(raw_dataset, Mapping):
            return False
        keys = set(raw_dataset)
        return "train" in keys and "test" in keys and bool({"validation", "valid"} & keys)

    @staticmethod
    def _access_plan(raw_dataset: Any) -> Any:
        if isinstance(raw_dataset, Mapping):
            plans = {
                name: getattr(dataset, "access_plan", None)
                for name, dataset in raw_dataset.items()
            }
            return {name: plan for name, plan in plans.items() if plan is not None} or None
        return getattr(raw_dataset, "access_plan", None)

    def prepare(self, raw_dataset: Any) -> tuple[Any, dict[str, Any], Any, str]:
        if self._is_split_mapping(raw_dataset):
            adapted = {name: self._adapt_dataset(dataset) for name, dataset in raw_dataset.items()}
            validation = adapted.get("validation", adapted.get("valid"))
            split_datasets = {
                "train": adapted["train"],
                "validation": validation,
                "test": adapted["test"],
            }
            sample_dataset = split_datasets["train"]
            loaders = make_split_dataloaders(
                split_datasets,
                batch_size=self.batch_size,
                seed=self.split_seed,
            )
            split_policy = "provider_declared"
        else:
            sample_dataset = self._adapt_dataset(raw_dataset)
            loaders = make_dataloaders(
                sample_dataset,
                batch_size=self.batch_size,
                seed=self.split_seed,
            )
            split_policy = "deterministic_random"

        if len(sample_dataset) < 1:
            raise ValueError("The dataset is empty")
        sample = sample_dataset[0]
        model, plan = load_model(
            self.model_id,
            dataset=self.dataset_id,
            sample=sample,
            task=self.task,
            overrides=self.model_overrides,
            return_plan=True,
        )
        return model, loaders, plan, split_policy

    def run(
        self,
        raw_dataset: Any,
        *,
        velocity_metrics: bool = False,
        metric_suites: str | Sequence[str] | None = None,
        metric_context: MetricContext | None = None,
    ) -> ExperimentResult:
        dataset_access = self._access_plan(raw_dataset)
        model, loaders, plan, split_policy = self.prepare(raw_dataset)
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
        selected_context = metric_context or self.metric_context
        metrics = trainer.evaluate(
            loaders["test"],
            velocity=velocity_metrics,
            metric_suites=selected_suites,
            metric_context=selected_context,
            include_metric_records=True,
        )
        manifest_path = (
            self._write_manifest(
                plan,
                training,
                metrics,
                split_policy=split_policy,
                metric_suites=selected_suites,
                metric_context=selected_context,
                dataset_access=dataset_access,
            )
            if self.output_dir
            else None
        )
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
        token: str | bool | None = None,
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

    def load_official_splits(
        self,
        *,
        streaming: bool = False,
        local_path: str | None = None,
        token: str | bool | None = None,
        adapt: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load provider-declared train/validation/test datasets without re-splitting."""

        spec = Catalog.load_builtin().dataset(self.dataset_id)
        if not spec.official_splits:
            raise ValueError(f"Dataset {self.dataset_id!r} does not declare official splits")
        split_aliases = {"validation": "valid"} if "valid" in spec.official_splits else {}
        datasets: dict[str, Any] = {}
        for canonical in ("train", "validation", "test"):
            provider_split = split_aliases.get(canonical, canonical)
            if provider_split not in spec.official_splits:
                raise ValueError(
                    f"Dataset {self.dataset_id!r} does not provide required split {provider_split!r}"
                )
            datasets[canonical] = self.load_dataset(
                split=provider_split,
                streaming=streaming,
                local_path=local_path,
                token=token,
                adapt=adapt,
                **kwargs,
            )
        return datasets

    def load_huggingface(
        self,
        *,
        split: str | None = None,
        config: str | None = None,
        streaming: bool = False,
        revision: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> Any:
        dataset_spec = Catalog.load_builtin().dataset(self.dataset_id)
        if dataset_spec.provider != "huggingface":
            raise ValueError(
                f"Dataset {self.dataset_id!r} uses provider {dataset_spec.provider!r}; "
                "call load_dataset() so NAVIER-CFD can route it correctly."
            )
        return HuggingFaceDatasetManager(token=token).load(
            dataset_spec,
            split=split,
            config=config,
            streaming=streaming,
            revision=revision,
            **kwargs,
        )

    def _write_manifest(
        self,
        plan: Any,
        training: TrainingResult,
        metrics: Mapping[str, Any],
        *,
        split_policy: str,
        metric_suites: str | Sequence[str],
        metric_context: MetricContext,
        dataset_access: Any = None,
    ) -> str:
        import json

        directory = Path(self.output_dir or ".")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "experiment-manifest.json"
        suite_names = [metric_suites] if isinstance(metric_suites, str) else list(metric_suites)
        payload = {
            "schema": "navier-cfd.experiment/v4",
            "dataset_id": self.dataset_id,
            "dataset_configuration": self.dataset_configuration,
            "dataset_provider": Catalog.load_builtin().dataset(self.dataset_id).provider,
            "dataset_access": dataset_access,
            "split_policy": split_policy,
            "model_id": self.model_id,
            "task": self.task.to_dict(),
            "adapter_options": dict(self.adapter_options),
            "model_plan": plan.to_dict(),
            "trainer": asdict(self.trainer_config),
            "split_seed": self.split_seed,
            "metric_suites": suite_names,
            "metric_context": asdict(metric_context),
            "training": {
                "best_epoch": training.best_epoch,
                "best_validation_loss": training.best_validation_loss,
                "checkpoint": training.checkpoint,
            },
            "metrics": dict(metrics),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)
