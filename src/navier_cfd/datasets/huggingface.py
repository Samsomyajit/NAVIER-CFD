from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..specs import DatasetSpec


class MissingDependencyError(RuntimeError):
    pass


@dataclass
class DownloadResult:
    repo_id: str
    local_path: str
    revision: str | None
    allow_patterns: tuple[str, ...]


class HuggingFaceDatasetManager:
    """Hugging Face dataset discovery, download, and streaming.

    Heavy imports are lazy so the rest of the registry can be used without network packages.
    """

    def __init__(self, token: str | None = None, endpoint: str | None = None) -> None:
        self.token = token
        self.endpoint = endpoint

    @staticmethod
    def _hub():
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the Hugging Face dependencies with `pip install navier-cfd`."
            ) from exc
        return HfApi, snapshot_download

    @staticmethod
    def _datasets():
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the datasets package with `pip install navier-cfd`."
            ) from exc
        return load_dataset

    def discover(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        HfApi, _ = self._hub()
        api = HfApi(token=self.token, endpoint=self.endpoint)
        rows = []
        for item in api.list_datasets(search=query, limit=limit, full=False):
            rows.append(
                {
                    "id": item.id,
                    "downloads": getattr(item, "downloads", None),
                    "likes": getattr(item, "likes", None),
                    "tags": list(getattr(item, "tags", []) or []),
                    "last_modified": str(getattr(item, "last_modified", "")),
                }
            )
        return rows

    def list_files(self, repo_id: str, revision: str | None = None) -> list[str]:
        HfApi, _ = self._hub()
        api = HfApi(token=self.token, endpoint=self.endpoint)
        return list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))

    def download(
        self,
        dataset: DatasetSpec | str,
        local_dir: str | Path,
        revision: str | None = None,
        allow_patterns: Iterable[str] | None = None,
        ignore_patterns: Iterable[str] | None = None,
    ) -> DownloadResult:
        _, snapshot_download = self._hub()
        repo_id = dataset.hf_repo_id if isinstance(dataset, DatasetSpec) else dataset
        if not repo_id:
            raise ValueError("This dataset card has no Hugging Face repository id.")
        allow = tuple(allow_patterns or ())
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=list(allow) or None,
            ignore_patterns=list(ignore_patterns or ()) or None,
            token=self.token,
            endpoint=self.endpoint,
        )
        return DownloadResult(repo_id=repo_id, local_path=path, revision=revision, allow_patterns=allow)

    def load(
        self,
        dataset: DatasetSpec | str,
        split: str | None = None,
        config: str | None = None,
        streaming: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> Any:
        load_dataset = self._datasets()
        repo_id = dataset.hf_repo_id if isinstance(dataset, DatasetSpec) else dataset
        if not repo_id:
            raise ValueError("This dataset card has no Hugging Face repository id.")
        if isinstance(dataset, DatasetSpec) and config is None:
            config = dataset.hf_config
        return load_dataset(
            repo_id,
            name=config,
            split=split,
            streaming=streaming,
            revision=revision,
            token=self.token,
            **kwargs,
        )
