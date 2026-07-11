"""NAVIER-CFD public API: neural and agentic workflows for computational fluid dynamics."""

from .catalogs import Catalog
from .recommender import Recommendation, recommend_models
from .specs import DatasetSpec, ModelSpec, TaskSpec

__all__ = [
    "Catalog",
    "DatasetSpec",
    "ModelSpec",
    "Recommendation",
    "TaskSpec",
    "recommend_models",
]

__version__ = "0.1.0"
