from .apebench import APEBenchDatasetManager, MissingAPEBenchDependency
from .cfdbench import (
    CFDBENCH_REPO,
    CFDBENCH_SCENARIOS,
    CFDBenchDatasetManager,
    load_cfdbench_archive_samples,
)
from .common import ListCFDDataset, ScientificDatasetAccessPlan
from .external import (
    LOCAL_DATASET_CONTRACTS,
    SUPPORTED_SUFFIXES,
    LocalDatasetContract,
    LocalDatasetProbe,
    LocalScientificDatasetManager,
)
from .pdebench import (
    MissingPDEBenchDependency,
    PDEBENCH_REPOSITORIES,
    PDEBenchDatasetManager,
    PDEBenchHDF5Dataset,
)
from .realpdebench import (
    REALPDEBENCH_REPO,
    REALPDEBENCH_SCENARIOS,
    RealPDEBenchDatasetManager,
    RealPDEBenchTrajectoryDataset,
)
from .upstream import (
    OFFICIAL_DATASET_SOURCES,
    OfficialDatasetAccessError,
    OfficialDatasetManager,
    OfficialDatasetSource,
    OfficialDownloadResult,
    OfficialUpstreamProbe,
)

# Keep local-loader error messages and provenance aligned with the verified publisher registry.
for _dataset_id, _source in OFFICIAL_DATASET_SOURCES.items():
    if _dataset_id not in LOCAL_DATASET_CONTRACTS:
        continue
    _contract = LOCAL_DATASET_CONTRACTS[_dataset_id]
    LOCAL_DATASET_CONTRACTS[_dataset_id] = LocalDatasetContract(
        dataset_id=_contract.dataset_id,
        provider=_contract.provider,
        source_url=_source.homepage,
        access_mode=_contract.access_mode,
        representation=_contract.representation,
        temporal=_contract.temporal,
        default_target_fields=_contract.default_target_fields,
        notes=_contract.notes + ("Stage official files with OfficialDatasetManager before loading.",),
    )

__all__ = [
    "APEBenchDatasetManager",
    "CFDBENCH_REPO",
    "CFDBENCH_SCENARIOS",
    "CFDBenchDatasetManager",
    "LOCAL_DATASET_CONTRACTS",
    "ListCFDDataset",
    "LocalDatasetContract",
    "LocalDatasetProbe",
    "LocalScientificDatasetManager",
    "MissingAPEBenchDependency",
    "MissingPDEBenchDependency",
    "OFFICIAL_DATASET_SOURCES",
    "OfficialDatasetAccessError",
    "OfficialDatasetManager",
    "OfficialDatasetSource",
    "OfficialDownloadResult",
    "OfficialUpstreamProbe",
    "PDEBENCH_REPOSITORIES",
    "PDEBenchDatasetManager",
    "PDEBenchHDF5Dataset",
    "REALPDEBENCH_REPO",
    "REALPDEBENCH_SCENARIOS",
    "RealPDEBenchDatasetManager",
    "RealPDEBenchTrajectoryDataset",
    "SUPPORTED_SUFFIXES",
    "ScientificDatasetAccessPlan",
    "load_cfdbench_archive_samples",
]
