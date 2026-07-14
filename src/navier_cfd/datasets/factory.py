from __future__ import annotations

from pathlib import Path
from typing import Any

from ..catalogs import Catalog
from .auth import resolve_hf_auth
from .huggingface import HuggingFaceDatasetManager
from .providers import (
    APEBenchDatasetManager,
    CFDBenchDatasetManager,
    LOCAL_DATASET_CONTRACTS,
    LocalScientificDatasetManager,
    PDEBenchDatasetManager,
    RealPDEBenchDatasetManager,
)
from .the_well import TheWellDatasetManager


def load_cfd_dataset(
    dataset_id: str,
    *,
    configuration: str | None = None,
    split: str = "train",
    streaming: bool = False,
    local_path: str | Path | None = None,
    token: str | bool | None = None,
    endpoint: str | None = None,
    adapt: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a registered dataset through its declared provider backend.

    Authentication resolves as explicit ``token`` > ``HF_TOKEN`` > the credential saved
    by ``hf auth login`` > anonymous. Scientific Hub repositories use provider-specific
    HDF5, archive, or Arrow loaders. Procedural and licensed/local datasets use their
    official Python API or secure local adapters and still return canonical ``CFDSample``.
    """

    spec = Catalog.load_builtin().dataset(dataset_id)
    if spec.provider == "the_well":
        if not configuration:
            raise ValueError(
                "The Well requires configuration=<well_dataset_name>, for example "
                "configuration='active_matter'."
            )
        auth = resolve_hf_auth(token)
        storage_options = dict(kwargs.pop("storage_options", {}) or {})
        if auth.token:
            storage_options.setdefault("token", auth.token)
        return TheWellDatasetManager().load(
            configuration,
            split=split,
            streaming=streaming,
            base_path=local_path,
            adapt=adapt,
            storage_options=storage_options or None,
            **kwargs,
        )
    if spec.provider == "pdebench":
        if not configuration:
            raise ValueError("PDEBench requires configuration='burgers' or 'advection'.")
        if not adapt:
            raise ValueError("PDEBench scientific HDF5 loading always returns canonical CFDSample objects")
        return PDEBenchDatasetManager(token=token, endpoint=endpoint).load(
            configuration,
            split=split,
            cache_dir=local_path,
            **kwargs,
        )
    if spec.provider == "cfdbench":
        if not configuration:
            raise ValueError("CFDBench requires configuration=cavity, tube, dam, or cylinder.")
        if not adapt:
            raise ValueError("CFDBench archive loading always returns canonical CFDSample objects")
        return CFDBenchDatasetManager(token=token, endpoint=endpoint).load(
            configuration,
            split=split,
            cache_dir=local_path,
            **kwargs,
        )
    if spec.provider == "realpdebench":
        if not configuration:
            raise ValueError(
                "RealPDEBench requires configuration=cylinder, controlled_cylinder, fsi, foil, or combustion."
            )
        if not adapt:
            raise ValueError("RealPDEBench Arrow loading always returns canonical CFDSample objects")
        return RealPDEBenchDatasetManager(token=token, endpoint=endpoint).load(
            configuration,
            split=split,
            cache_dir=local_path,
            **kwargs,
        )
    if spec.provider == "apebench":
        if not configuration:
            raise ValueError(
                "APEBench requires a scenario configuration, for example configuration='Advection'."
            )
        if local_path is not None:
            raise ValueError(
                "APEBench is generated procedurally and does not use local_path. "
                "Pass scenario_kwargs and sample/window limits instead."
            )
        if streaming:
            raise ValueError("APEBench procedural generation does not support streaming=True")
        if not adapt:
            raise ValueError("APEBench generation always returns canonical CFDSample objects")
        return APEBenchDatasetManager().load(
            configuration,
            split=split,
            **kwargs,
        )
    if spec.provider in LOCAL_DATASET_CONTRACTS:
        if local_path is None:
            contract = LOCAL_DATASET_CONTRACTS[spec.provider]
            raise ValueError(
                f"{spec.name} requires local_path pointing to an official local export. "
                f"Obtain the data from {contract.source_url} and probe it before loading."
            )
        if streaming:
            raise ValueError(f"{spec.name} local provider does not support streaming=True")
        if not adapt:
            raise ValueError(f"{spec.name} local loading always returns canonical CFDSample objects")
        return LocalScientificDatasetManager().load(
            spec.provider,
            local_path=local_path,
            configuration=configuration,
            split=split,
            **kwargs,
        )
    if spec.provider == "huggingface":
        if local_path is not None:
            raise ValueError(
                "local_path is provider-specific; use HuggingFaceDatasetManager.download "
                "for local snapshots."
            )
        return HuggingFaceDatasetManager(token=token, endpoint=endpoint).load(
            spec,
            split=split,
            config=configuration,
            streaming=streaming,
            **kwargs,
        )
    raise ValueError(
        f"Dataset {dataset_id!r} uses provider {spec.provider!r}, which has no built-in "
        "runtime loader."
    )


__all__ = ["load_cfd_dataset"]
