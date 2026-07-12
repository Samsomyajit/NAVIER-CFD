"""NAVIER-CFD public API: neural and agentic workflows for computational fluid dynamics."""

from .catalogs import Catalog
from .evidence import (
    ALGORITHM_VERSION,
    EvidenceMatch,
    EvidenceRecord,
    EvidenceSummary,
    load_builtin_evidence,
    score_model_evidence,
)
from .models import (
    ExternalInstallDisabledError,
    ExternalModelRecipe,
    ModelDependencyError,
    ModelHandle,
    ModelHub,
    ModelHubError,
    ModelNotExecutableError,
    ModelStatus,
    UnknownModelError,
    get_model_hub,
    install_model,
    list_models,
    load_model,
    model_info,
)
from .recommender import Recommendation, recommend_models
from .specs import DatasetSpec, ModelSpec, TaskSpec

__all__ = [
    "ALGORITHM_VERSION",
    "Catalog",
    "DatasetSpec",
    "EvidenceMatch",
    "EvidenceRecord",
    "EvidenceSummary",
    "ExternalInstallDisabledError",
    "ExternalModelRecipe",
    "ModelDependencyError",
    "ModelHandle",
    "ModelHub",
    "ModelHubError",
    "ModelNotExecutableError",
    "ModelSpec",
    "ModelStatus",
    "Recommendation",
    "TaskSpec",
    "UnknownModelError",
    "get_model_hub",
    "install_model",
    "list_models",
    "load_builtin_evidence",
    "load_model",
    "model_info",
    "recommend_models",
    "score_model_evidence",
]

__version__ = "0.3.0"
