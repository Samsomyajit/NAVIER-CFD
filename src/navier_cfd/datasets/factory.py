from __future__ import annotations

from pathlib import Path
from typing import Any

from ..catalogs import Catalog
from .huggingface import HuggingFaceDatasetManager
from .the_well import TheWellDatasetManager


def load_cfd_dataset(
    dataset_id: str,
    *,
    configuration: str | None = None,
    split: str = "train",
    streaming: bool = False,
    local_path: str | Path | None = None,
    token: str | None = None,
    adapt: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a registered dataset through its declared provider backend.

    ``configuration`` means a provider-native dataset/configuration name. For The Well,
    it is the required ``well_dataset_name``. Generic Hugging Face datasets continue to
    use their repository ID and optional ``hf_config``.
    """

    spec = Catalog.load_builtin().dataset(dataset_id)
    if spec.provider == "the_well":
        if not configuration:
            raise ValueError(
                "The Well requires configuration=<well_dataset_name>, for example "
                "configuration='active_matter'."
            )
        return TheWellDatasetManager().load(
            configuration,
            split=split,
            streaming=streaming,
            base_path=local_path,
            adapt=adapt,
            **kwargs,
        )
    if spec.provider == "huggingface":
        if local_path is not None:
            raise ValueError(
                "local_path is provider-specific; use HuggingFaceDatasetManager.download "
                "for local snapshots."
            )
        return HuggingFaceDatasetManager(token=token).load(
            spec,
            split=split,
            config=configuration,
            streaming=streaming,
            **kwargs,
        )
    raise ValueError(
        f"Dataset {dataset_id!r} uses provider {spec.provider!r}, which has no built-in "
        "runtime loader yet. Use the dataset's official loader and NAVIER-CFD adapter."
    )


__all__ = ["load_cfd_dataset"]
