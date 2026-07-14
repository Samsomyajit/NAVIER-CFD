from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..specs import DatasetSpec
from .auth import HuggingFaceAuth, resolve_hf_auth


class MissingDependencyError(RuntimeError):
    pass


class HuggingFaceDatasetError(RuntimeError):
    """Base error for provider-aware Hugging Face access."""


class HuggingFaceAuthenticationError(HuggingFaceDatasetError):
    pass


class HuggingFaceRepositoryError(HuggingFaceDatasetError):
    pass


class HuggingFaceNetworkError(HuggingFaceDatasetError):
    pass


class UnsupportedHuggingFaceLayoutError(HuggingFaceDatasetError):
    pass


@dataclass(frozen=True)
class DownloadResult:
    repo_id: str
    local_path: str
    revision: str | None
    allow_patterns: tuple[str, ...]
    resolved_revision: str | None = None
    auth_source: str = "anonymous"


@dataclass(frozen=True)
class DatasetProbe:
    repo_id: str
    reachable: bool
    authenticated: bool
    auth_source: str
    revision: str | None
    resolved_revision: str | None
    files: tuple[str, ...]
    detected_layout: str
    recommended_strategy: str
    error_category: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_hf_layout(files: Sequence[str]) -> tuple[str, str]:
    """Classify repository storage without assuming ``datasets.load_dataset`` works."""

    lowered = tuple(path.lower() for path in files)
    has_arrow = any(path.endswith(".arrow") for path in lowered)
    has_saved_dataset_metadata = any(
        path.endswith("state.json") or path.endswith("dataset_info.json") for path in lowered
    )
    if has_arrow and has_saved_dataset_metadata:
        return "saved_arrow", "snapshot_arrow"
    if any(path.endswith((".h5", ".hdf5")) for path in lowered):
        return "scientific_hdf5", "snapshot_hdf5"
    if any(path.endswith((".zip", ".tar", ".tar.gz", ".tgz")) for path in lowered):
        return "archive", "snapshot_archive"
    if any(path.endswith(".parquet") for path in lowered):
        return "parquet", "datasets"
    if any(path.endswith((".jsonl", ".json", ".csv")) for path in lowered):
        return "tabular", "datasets"
    if any(path.endswith(".py") for path in lowered):
        return "dataset_script", "datasets"
    return "unknown", "inspect"


class HuggingFaceDatasetManager:
    """Authenticated Hugging Face discovery, probing, download, and loading.

    Credentials resolve as explicit token, ``HF_TOKEN``, stored ``hf auth login``
    credential, then anonymous access. Provider-native and scientific-layout datasets
    are deliberately routed through dedicated adapters.
    """

    def __init__(
        self,
        token: str | bool | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.auth: HuggingFaceAuth = resolve_hf_auth(token)
        self.token = self.auth.token
        self.endpoint = endpoint

    @staticmethod
    def _hub() -> tuple[Any, Any, Any]:
        try:
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the Hugging Face dependencies with `pip install navier-cfd`."
            ) from exc
        return HfApi, hf_hub_download, snapshot_download

    @staticmethod
    def _datasets() -> tuple[Any, Any]:
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:
            raise MissingDependencyError(
                "Install the datasets package with `pip install navier-cfd`."
            ) from exc
        return load_dataset, load_from_disk

    @staticmethod
    def _repo_id(dataset: DatasetSpec | str) -> str:
        if isinstance(dataset, str):
            return dataset
        if dataset.provider != "huggingface":
            if dataset.provider == "the_well":
                raise ValueError(
                    "The Well is not one datasets.load_dataset repository. "
                    "Use TheWellDatasetManager.load(dataset_name=...) or "
                    "load_cfd_dataset('the_well', configuration=...)."
                )
            raise ValueError(
                f"Dataset {dataset.id!r} uses provider {dataset.provider!r}; "
                "use its provider-specific loader."
            )
        if not dataset.hf_repo_id:
            raise ValueError("This dataset card has no Hugging Face repository id.")
        return dataset.hf_repo_id

    @staticmethod
    def _error_category(exc: Exception) -> str:
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        if "gatedrepo" in name or "401" in message or "403" in message:
            return "authentication"
        if "repositorynotfound" in name or "entrynotfound" in name or "404" in message:
            return "not_found"
        if any(token in name for token in ("timeout", "connection", "network")):
            return "network"
        if any(token in message for token in ("timed out", "connection", "network")):
            return "network"
        return "hub_error"

    def _api(self) -> Any:
        HfApi, _, _ = self._hub()
        return HfApi(token=self.token, endpoint=self.endpoint)

    def auth_status(self, *, verify: bool = True) -> dict[str, Any]:
        status: dict[str, Any] = {
            "authenticated": self.auth.authenticated,
            "source": self.auth.source,
            "endpoint": self.endpoint or "https://huggingface.co",
            "user": None,
            "verified": False,
        }
        if not verify or not self.auth.authenticated:
            return status
        try:
            identity = self._api().whoami(token=self.token)
            status["user"] = identity.get("name") or identity.get("fullname")
            status["verified"] = True
        except Exception as exc:
            status["error_category"] = self._error_category(exc)
            status["message"] = str(exc)
        return status

    def discover(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = []
        for item in self._api().list_datasets(search=query, limit=limit, full=False):
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

    def dataset_info(self, repo_id: str, revision: str | None = None) -> Any:
        return self._api().dataset_info(repo_id=repo_id, revision=revision, token=self.token)

    def resolve_revision(self, repo_id: str, revision: str | None = None) -> str | None:
        return getattr(self.dataset_info(repo_id, revision=revision), "sha", revision)

    def list_files(self, repo_id: str, revision: str | None = None) -> list[str]:
        return list(
            self._api().list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=self.token,
            )
        )

    def list_file_entries(self, repo_id: str, revision: str | None = None) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        api = self._api()
        try:
            iterator = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                recursive=True,
                expand=True,
                token=self.token,
            )
            for item in iterator:
                path = getattr(item, "path", None)
                if not path or item.__class__.__name__.lower().endswith("folder"):
                    continue
                entries.append(
                    {
                        "path": path,
                        "size": getattr(item, "size", None),
                        "blob_id": getattr(item, "blob_id", None),
                    }
                )
        except TypeError:
            return [
                {"path": path, "size": None, "blob_id": None}
                for path in self.list_files(repo_id, revision)
            ]
        return entries

    def probe(
        self,
        dataset: DatasetSpec | str,
        *,
        revision: str | None = None,
        max_files: int = 200,
    ) -> DatasetProbe:
        repo_id = self._repo_id(dataset)
        try:
            info = self.dataset_info(repo_id, revision=revision)
            files = tuple(self.list_files(repo_id, revision=revision))
            layout, strategy = classify_hf_layout(files)
            return DatasetProbe(
                repo_id=repo_id,
                reachable=True,
                authenticated=self.auth.authenticated,
                auth_source=self.auth.source,
                revision=revision,
                resolved_revision=getattr(info, "sha", None),
                files=files[:max_files],
                detected_layout=layout,
                recommended_strategy=strategy,
            )
        except Exception as exc:
            category = self._error_category(exc)
            return DatasetProbe(
                repo_id=repo_id,
                reachable=False,
                authenticated=self.auth.authenticated,
                auth_source=self.auth.source,
                revision=revision,
                resolved_revision=None,
                files=(),
                detected_layout="unavailable",
                recommended_strategy="authenticate" if category == "authentication" else "inspect",
                error_category=category,
                message=str(exc),
            )

    def download_file(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
    ) -> Path:
        _, hf_hub_download, _ = self._hub()
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_dir=str(local_dir) if local_dir is not None else None,
                token=self.token,
                endpoint=self.endpoint,
            )
        except Exception as exc:
            category = self._error_category(exc)
            if category == "authentication":
                raise HuggingFaceAuthenticationError(str(exc)) from exc
            if category == "not_found":
                raise HuggingFaceRepositoryError(str(exc)) from exc
            if category == "network":
                raise HuggingFaceNetworkError(str(exc)) from exc
            raise HuggingFaceDatasetError(str(exc)) from exc
        return Path(path)

    def download(
        self,
        dataset: DatasetSpec | str,
        local_dir: str | Path,
        revision: str | None = None,
        allow_patterns: Iterable[str] | None = None,
        ignore_patterns: Iterable[str] | None = None,
    ) -> DownloadResult:
        repo_id = self._repo_id(dataset)
        _, _, snapshot_download = self._hub()
        allow = tuple(allow_patterns or ())
        resolved = None
        try:
            resolved = self.resolve_revision(repo_id, revision)
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
        except Exception as exc:
            category = self._error_category(exc)
            if category == "authentication":
                raise HuggingFaceAuthenticationError(str(exc)) from exc
            if category == "not_found":
                raise HuggingFaceRepositoryError(str(exc)) from exc
            if category == "network":
                raise HuggingFaceNetworkError(str(exc)) from exc
            raise HuggingFaceDatasetError(str(exc)) from exc
        return DownloadResult(
            repo_id=repo_id,
            local_path=path,
            revision=revision,
            allow_patterns=allow,
            resolved_revision=resolved,
            auth_source=self.auth.source,
        )

    def load(
        self,
        dataset: DatasetSpec | str,
        split: str | None = None,
        config: str | None = None,
        streaming: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> Any:
        repo_id = self._repo_id(dataset)
        load_dataset, _ = self._datasets()
        if isinstance(dataset, DatasetSpec) and config is None:
            config = dataset.hf_config
        try:
            return load_dataset(
                repo_id,
                name=config,
                split=split,
                streaming=streaming,
                revision=revision,
                token=self.token,
                **kwargs,
            )
        except Exception as exc:
            message = str(exc)
            lower = message.lower()
            if any(token in lower for token in ("hdf5", "arrow", "zip", "archive")):
                raise UnsupportedHuggingFaceLayoutError(
                    f"Repository {repo_id!r} is reachable but is not a standard datasets.load_dataset "
                    "layout. Use the registered provider-specific loader. Original error: "
                    f"{message}"
                ) from exc
            raise


__all__ = [
    "DatasetProbe",
    "DownloadResult",
    "HuggingFaceAuthenticationError",
    "HuggingFaceDatasetError",
    "HuggingFaceDatasetManager",
    "HuggingFaceNetworkError",
    "HuggingFaceRepositoryError",
    "MissingDependencyError",
    "UnsupportedHuggingFaceLayoutError",
    "classify_hf_layout",
]
