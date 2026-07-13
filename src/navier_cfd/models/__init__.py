from .base import ExternalModelAdapter, ModelAdapter, ModelFactory
from .config import (
    DATASET_MODEL_DEFAULTS,
    DatasetModelDefaults,
    ModelBuildPlan,
    translate_model_config,
)
from .native import MissingTorchDependency, NATIVE_BUILDERS, build_deeponet, build_fno, build_pinn
from .native_latent import LATENT_BUILDERS, build_latent_reference
from .native_suite import NATIVE_REFERENCE_FAMILIES, REFERENCE_BUILDERS, build_native_reference
from .pibert import PIBERT, build_pibert

NATIVE_BUILDERS.setdefault("pibert", build_pibert)
NATIVE_BUILDERS.update({model_id: builder for model_id, builder in REFERENCE_BUILDERS.items() if model_id not in NATIVE_BUILDERS})
NATIVE_BUILDERS.update(LATENT_BUILDERS)

from .hub import (  # noqa: E402
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
    model_info,
)
from .dataset_factory import configure_model_for_dataset, load_model  # noqa: E402
from .conformance import AdapterConformanceReport, validate_model_adapter  # noqa: E402

__all__ = [
    "AdapterConformanceReport",
    "DATASET_MODEL_DEFAULTS",
    "DatasetModelDefaults",
    "ExternalInstallDisabledError",
    "ExternalModelAdapter",
    "ExternalModelRecipe",
    "LATENT_BUILDERS",
    "MissingTorchDependency",
    "ModelAdapter",
    "ModelBuildPlan",
    "ModelDependencyError",
    "ModelFactory",
    "ModelHandle",
    "ModelHub",
    "ModelHubError",
    "ModelNotExecutableError",
    "ModelStatus",
    "NATIVE_REFERENCE_FAMILIES",
    "PIBERT",
    "UnknownModelError",
    "build_deeponet",
    "build_fno",
    "build_latent_reference",
    "build_native_reference",
    "build_pibert",
    "build_pinn",
    "configure_model_for_dataset",
    "get_model_hub",
    "install_model",
    "list_models",
    "load_model",
    "model_info",
    "translate_model_config",
    "validate_model_adapter",
]
