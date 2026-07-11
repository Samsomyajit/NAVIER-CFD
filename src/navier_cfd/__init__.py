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
from .recommender import Recommendation, recommend_models
from .specs import DatasetSpec, ModelSpec, TaskSpec

__all__ = [
    "ALGORITHM_VERSION",
    "Catalog",
    "DatasetSpec",
    "EvidenceMatch",
    "EvidenceRecord",
    "EvidenceSummary",
    "ModelSpec",
    "Recommendation",
    "TaskSpec",
    "load_builtin_evidence",
    "recommend_models",
    "score_model_evidence",
]

__version__ = "0.2.0"
