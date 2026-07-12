from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.util
import inspect
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping

from ..catalogs import Catalog
from ..specs import ModelSpec, TaskSpec
from .native import MissingTorchDependency, NATIVE_BUILDERS


class ModelHubError(RuntimeError):
    """Base exception for executable model-hub operations."""


class UnknownModelError(ModelHubError):
    """Raised when a model identifier is not registered in NAVIER-CFD."""


class ModelNotExecutableError(ModelHubError):
    """Raised when a model has metadata but no executable adapter."""


class ModelDependencyError(ModelHubError):
    """Raised when an executable adapter is known but its dependency is absent."""


class ExternalInstallDisabledError(ModelHubError):
    """Raised when code installation was requested without explicit opt-in."""


@dataclass(frozen=True)
class ExternalModelRecipe:
    """Import recipe for an upstream model implementation.

    ``entrypoint`` follows ``package.module:Object`` syntax. ``install_spec`` is
    passed as one argument to ``python -m pip install`` only after the caller
    explicitly enables external installation.
    """

    model_id: str
    entrypoint: str
    install_spec: str | None = None
    repository: str | None = None
    notes: str = ""

    @property
    def module(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def object_path(self) -> str:
        parts = self.entrypoint.split(":", 1)
        return parts[1] if len(parts) == 2 else ""


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    name: str
    mode: str
    executable: bool
    dependency_available: bool
    installable: bool
    entrypoint: str | None
    install_spec: str | None
    repository: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelHandle:
    """A lightweight handle for one model registered under the common API."""

    def __init__(self, hub: "ModelHub", spec: ModelSpec) -> None:
        self._hub = hub
        self.spec = spec

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def status(self) -> ModelStatus:
        return self._hub.status(self.spec.id)

    def load(self, *args: Any, task: TaskSpec | None = None, **kwargs: Any) -> Any:
        return self._hub.load(self.spec.id, *args, task=task, **kwargs)

    def install(
        self,
        *,
        allow_external: bool = False,
        upgrade: bool = False,
        pip_args: Iterable[str] = (),
    ) -> None:
        self._hub.install(
            self.spec.id,
            allow_external=allow_external,
            upgrade=upgrade,
            pip_args=pip_args,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.spec.to_dict()
        payload["runtime"] = self.status.to_dict()
        return payload


class ModelHub:
    """Unified registry and lazy loader for NAVIER-CFD model implementations.

    Every catalog model receives a ``ModelHandle``. Native reference models are
    constructed directly. Upstream projects can be connected through an explicit
    entrypoint recipe without importing or installing third-party code at package
    import time.
    """

    def __init__(
        self,
        catalog: Catalog | None = None,
        *,
        native_builders: Mapping[str, Callable[..., Any]] | None = None,
        external_recipes: Iterable[ExternalModelRecipe] = (),
    ) -> None:
        self.catalog = catalog or Catalog.load_builtin()
        self._native_builders: dict[str, Callable[..., Any]] = dict(native_builders or NATIVE_BUILDERS)
        self._external_recipes: dict[str, ExternalModelRecipe] = {}
        for recipe in external_recipes:
            self.register_external_recipe(recipe)

    def ids(self) -> tuple[str, ...]:
        return tuple(model.id for model in self.catalog.models)

    def models(self) -> tuple[ModelHandle, ...]:
        return tuple(ModelHandle(self, model) for model in self.catalog.models)

    def model(self, model_id: str) -> ModelHandle:
        try:
            return ModelHandle(self, self.catalog.model(model_id))
        except KeyError as exc:
            raise UnknownModelError(str(exc)) from exc

    def register_builder(
        self,
        model_id: str,
        builder: Callable[..., Any],
        *,
        replace: bool = False,
    ) -> None:
        self.model(model_id)
        if model_id in self._native_builders and not replace:
            raise ValueError(f"A native builder is already registered for {model_id}")
        self._native_builders[model_id] = builder

    def register_external(
        self,
        model_id: str,
        *,
        entrypoint: str,
        install_spec: str | None = None,
        repository: str | None = None,
        notes: str = "",
        replace: bool = False,
    ) -> None:
        self.register_external_recipe(
            ExternalModelRecipe(
                model_id=model_id,
                entrypoint=entrypoint,
                install_spec=install_spec,
                repository=repository,
                notes=notes,
            ),
            replace=replace,
        )

    def register_external_recipe(self, recipe: ExternalModelRecipe, *, replace: bool = False) -> None:
        self.model(recipe.model_id)
        if ":" not in recipe.entrypoint:
            raise ValueError("entrypoint must use 'package.module:Object' syntax")
        if recipe.model_id in self._external_recipes and not replace:
            raise ValueError(f"An external recipe is already registered for {recipe.model_id}")
        self._external_recipes[recipe.model_id] = recipe

    def _module_available(self, module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            return False

    def status(self, model_id: str) -> ModelStatus:
        spec = self.model(model_id).spec
        if model_id in self._native_builders:
            torch_available = self._module_available("torch")
            return ModelStatus(
                model_id=model_id,
                name=spec.name,
                mode="native",
                executable=True,
                dependency_available=torch_available,
                installable=True,
                entrypoint=f"navier_cfd.models.native:{self._native_builders[model_id].__name__}",
                install_spec="navier-cfd[torch]",
                repository=spec.repository,
                message=(
                    "ready" if torch_available else "native builder available; install the torch extra"
                ),
            )

        recipe = self._external_recipes.get(model_id)
        if recipe is not None:
            available = self._module_available(recipe.module)
            return ModelStatus(
                model_id=model_id,
                name=spec.name,
                mode="external_adapter",
                executable=True,
                dependency_available=available,
                installable=bool(recipe.install_spec),
                entrypoint=recipe.entrypoint,
                install_spec=recipe.install_spec,
                repository=recipe.repository or spec.repository,
                message=(
                    "ready" if available else "adapter registered but its upstream dependency is not installed"
                ),
            )

        if spec.repository:
            return ModelStatus(
                model_id=model_id,
                name=spec.name,
                mode="source_available",
                executable=False,
                dependency_available=False,
                installable=False,
                entrypoint=None,
                install_spec=None,
                repository=spec.repository,
                message="official source is known, but no stable Python constructor has been registered",
            )

        return ModelStatus(
            model_id=model_id,
            name=spec.name,
            mode="metadata",
            executable=False,
            dependency_available=False,
            installable=False,
            entrypoint=None,
            install_spec=None,
            repository=None,
            message="catalog metadata is available; an executable adapter is still required",
        )

    def executable(self, *, ready_only: bool = False) -> tuple[ModelHandle, ...]:
        handles = []
        for handle in self.models():
            status = handle.status
            if status.executable and (status.dependency_available or not ready_only):
                handles.append(handle)
        return tuple(handles)

    def install(
        self,
        model_id: str,
        *,
        allow_external: bool = False,
        upgrade: bool = False,
        pip_args: Iterable[str] = (),
    ) -> None:
        status = self.status(model_id)
        install_spec = status.install_spec
        if not install_spec:
            repository = f" See {status.repository}." if status.repository else ""
            raise ModelNotExecutableError(
                f"{status.name} has no safe package installation recipe.{repository} "
                "Register one with ModelHub.register_external()."
            )
        if not allow_external:
            raise ExternalInstallDisabledError(
                "Installing model dependencies executes third-party packaging code. "
                "Repeat with allow_external=True after reviewing the source and license."
            )

        command = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            command.append("--upgrade")
        command.extend(str(arg) for arg in pip_args)
        command.append(install_spec)
        subprocess.run(command, check=True)

    def _resolve_entrypoint(self, entrypoint: str) -> Any:
        module_name, object_path = entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        target: Any = module
        for part in object_path.split("."):
            if part:
                target = getattr(target, part)
        return target

    def load(
        self,
        model_id: str,
        *args: Any,
        task: TaskSpec | None = None,
        entrypoint: str | None = None,
        install_spec: str | None = None,
        allow_install: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Construct a model through the common API.

        ``entrypoint`` can temporarily connect an installed upstream implementation
        using ``module:Object`` syntax. Passing ``install_spec`` never installs code
        unless ``allow_install=True`` is also supplied.
        """

        spec = self.model(model_id).spec
        if entrypoint is None and model_id in self._native_builders:
            try:
                return self._native_builders[model_id](task=task, spec=spec, *args, **kwargs)
            except MissingTorchDependency as exc:
                raise ModelDependencyError(str(exc)) from exc

        recipe = self._external_recipes.get(model_id)
        selected_entrypoint = entrypoint or (recipe.entrypoint if recipe else None)
        selected_install_spec = install_spec or (recipe.install_spec if recipe else None)

        if selected_entrypoint is None:
            repository = f" Official source: {spec.repository}." if spec.repository else ""
            raise ModelNotExecutableError(
                f"{spec.name} is registered, but NAVIER-CFD does not yet know a stable constructor."
                f"{repository} Connect it with register_external() or pass entrypoint='module:Object'."
            )

        module_name = selected_entrypoint.split(":", 1)[0]
        if not self._module_available(module_name):
            if allow_install and selected_install_spec:
                temporary = ExternalModelRecipe(
                    model_id=model_id,
                    entrypoint=selected_entrypoint,
                    install_spec=selected_install_spec,
                    repository=spec.repository,
                )
                previous = self._external_recipes.get(model_id)
                self._external_recipes[model_id] = temporary
                try:
                    self.install(model_id, allow_external=True)
                finally:
                    if previous is None:
                        self._external_recipes.pop(model_id, None)
                    else:
                        self._external_recipes[model_id] = previous
            else:
                command = f"pip install {selected_install_spec}" if selected_install_spec else "install its official package"
                raise ModelDependencyError(
                    f"Cannot import {module_name!r} for {spec.name}; {command}, then retry."
                )

        target = self._resolve_entrypoint(selected_entrypoint)
        if inspect.isclass(target) or callable(target):
            return target(*args, **kwargs)
        if args or kwargs:
            raise TypeError(f"Entrypoint {selected_entrypoint} resolves to an object, not a constructor")
        return target


_DEFAULT_HUB: ModelHub | None = None


def get_model_hub() -> ModelHub:
    global _DEFAULT_HUB
    if _DEFAULT_HUB is None:
        _DEFAULT_HUB = ModelHub()
    return _DEFAULT_HUB


def list_models() -> tuple[ModelHandle, ...]:
    return get_model_hub().models()


def model_info(model_id: str) -> dict[str, Any]:
    return get_model_hub().model(model_id).to_dict()


def load_model(
    model_id: str,
    *args: Any,
    task: TaskSpec | None = None,
    **kwargs: Any,
) -> Any:
    return get_model_hub().load(model_id, *args, task=task, **kwargs)


def install_model(
    model_id: str,
    *,
    allow_external: bool = False,
    upgrade: bool = False,
    pip_args: Iterable[str] = (),
) -> None:
    get_model_hub().install(
        model_id,
        allow_external=allow_external,
        upgrade=upgrade,
        pip_args=pip_args,
    )


__all__ = [
    "ExternalInstallDisabledError",
    "ExternalModelRecipe",
    "ModelDependencyError",
    "ModelHandle",
    "ModelHub",
    "ModelHubError",
    "ModelNotExecutableError",
    "ModelStatus",
    "UnknownModelError",
    "get_model_hub",
    "install_model",
    "list_models",
    "load_model",
    "model_info",
]
