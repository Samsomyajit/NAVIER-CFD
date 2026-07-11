"""Build the static NAVIER-CFD web recommender assets.

The generated browser runtime packages the same Python specs, catalog, evidence,
and recommender used by the library. Pyodide loads this zip in the website, so
the interactive recommendations are produced by the project implementation
rather than a separate opaque service.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "navier_cfd"
WEBSITE = ROOT / "website"
RUNTIME = WEBSITE / "runtime"
DATA = WEBSITE / "data"


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_catalog() -> dict[str, Any]:
    from navier_cfd import Catalog

    catalog = Catalog.load_builtin()
    return {
        "models": [jsonable(model) for model in catalog.models],
        "datasets": [jsonable(dataset) for dataset in catalog.datasets],
    }


BRIDGE = r'''from __future__ import annotations

import inspect
import json
import pkgutil
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints

from navier_cfd import Catalog, TaskSpec, recommend_models
from navier_cfd.evidence import EvidenceRecord
import navier_cfd.recommender as _recommender_module

_CATALOG = Catalog.load_builtin()
_EVIDENCE_BYTES = pkgutil.get_data("navier_cfd", "data/paper_evidence.json")
if _EVIDENCE_BYTES is None:
    raise RuntimeError("NAVIER-CFD paper evidence catalog is missing from the browser runtime")
_EVIDENCE = tuple(
    EvidenceRecord.from_dict(row)
    for row in json.loads(_EVIDENCE_BYTES.decode("utf-8"))
)
# The normal loader uses a filesystem Path. The browser package is zip-imported,
# so provide the same immutable evidence records directly to the recommender.
_recommender_module.load_builtin_evidence = lambda: _EVIDENCE


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _coerce(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None:
        if origin in (list, tuple, set, frozenset):
            item_type = args[0] if args else Any
            values = value if isinstance(value, (list, tuple, set)) else [value]
            converted = [_coerce(item_type, item) for item in values]
            return origin(converted) if origin is not tuple else tuple(converted)
        if origin is dict:
            return value
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _coerce(candidate, value)
            except (TypeError, ValueError):
                pass
        return value
    try:
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            return annotation(value)
    except TypeError:
        pass
    if annotation is bool:
        return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
    if annotation in (int, float, str):
        return annotation(value)
    return value


def _build_task(raw: dict[str, Any]) -> TaskSpec:
    aliases = {
        "conservation": "requires_conservation",
        "uncertainty": "requires_uncertainty",
        "long_rollout": "requires_long_rollout",
        "geometry_transfer": "requires_geometry_transfer",
        "mesh_transfer": "requires_mesh_transfer",
        "memory_gb": "hardware_memory_gb",
    }
    raw = {aliases.get(key, key): value for key, value in raw.items()}
    hints = get_type_hints(TaskSpec)
    kwargs: dict[str, Any] = {}
    for field in fields(TaskSpec):
        if field.name in raw and raw[field.name] not in ("", None):
            kwargs[field.name] = _coerce(hints.get(field.name, field.type), raw[field.name])
    return TaskSpec(**kwargs)


def recommend_json(task_json: str, top_k: int = 8) -> str:
    task = _build_task(json.loads(task_json))
    ranked = recommend_models(task, _CATALOG.models, top_k=int(top_k))
    return json.dumps([_jsonable(item) for item in ranked])


def health_json() -> str:
    return json.dumps({
        "engine": "navier_cfd.recommender",
        "models": len(_CATALOG.models),
        "datasets": len(_CATALOG.datasets),
        "evidence_records": len(_EVIDENCE),
        "status": "ready",
    })
'''


def write_runtime_zip() -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / "navier_runtime.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for source in sorted(PACKAGE.rglob("*")):
            if source.is_dir() or "__pycache__" in source.parts:
                continue
            if source.suffix not in {".py", ".json", ".yaml", ".yml", ".toml"}:
                continue
            relative = source.relative_to(PACKAGE.parent)
            if relative.as_posix() == "navier_cfd/__init__.py":
                zf.writestr(
                    "navier_cfd/__init__.py",
                    "from .catalogs import Catalog\n"
                    "from .specs import TaskSpec\n"
                    "from .recommender import recommend_models\n"
                    "__all__ = ['Catalog', 'TaskSpec', 'recommend_models']\n",
                )
            else:
                zf.write(source, relative.as_posix())
        zf.writestr("navier_web_bridge.py", BRIDGE)
    return archive


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    (DATA / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    archive = write_runtime_zip()
    print(f"Exported {len(catalog['models'])} models and {len(catalog['datasets'])} datasets")
    print(f"Runtime: {archive}")


if __name__ == "__main__":
    main()
