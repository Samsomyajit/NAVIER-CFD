from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from ..agents import AgentOrchestrator
from ..catalogs import Catalog
from ..figures import FigureSpec, audit_figure_spec
from ..metrics import METRICS, SUITES
from ..recommender import recommend_models
from ..specs import TaskSpec


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
            "annotations": {"readOnlyHint": self.read_only},
        }


class ToolRegistry:
    """Deterministic tool surface shared by MCP, Codex skills, and tests."""

    def __init__(self, catalog: Catalog | None = None) -> None:
        self.catalog = catalog or Catalog.load_builtin()
        self._tools: dict[str, tuple[ToolSpec, Callable[[Mapping[str, Any]], Any]]] = {}
        self._register_builtin_tools()

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool name: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self._tools.values())

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        try:
            _, handler = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown NAVIER-CFD tool {name!r}; available: {sorted(self._tools)}") from exc
        return handler(dict(arguments or {}))

    def _register_builtin_tools(self) -> None:
        empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
        self.register(
            ToolSpec("list_datasets", "List registered CFD/PDE datasets.", empty_schema),
            self._list_datasets,
        )
        self.register(
            ToolSpec("list_models", "List registered surrogate model families.", empty_schema),
            self._list_models,
        )
        self.register(
            ToolSpec(
                "plan_research",
                "Interpret a client CFD problem and build a deterministic NAVIER-CFD benchmark plan.",
                {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            ),
            self._plan_research,
        )
        self.register(
            ToolSpec(
                "recommend_models",
                "Rank compatible models for a structured CFD task specification.",
                {
                    "type": "object",
                    "properties": {
                        "task": {"type": "object"},
                        "top_k": {"type": "integer", "minimum": 1, "default": 8},
                        "evidence_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                },
            ),
            self._recommend_models,
        )
        self.register(
            ToolSpec("list_metric_suites", "List available numerical and physical metric suites.", empty_schema),
            self._list_metric_suites,
        )
        self.register(
            ToolSpec(
                "audit_figure_spec",
                "Audit a publication figure specification for scientific and visual integrity.",
                {
                    "type": "object",
                    "properties": {"spec": {"type": "object"}},
                    "required": ["spec"],
                    "additionalProperties": False,
                },
            ),
            self._audit_figure_spec,
        )

    def _list_datasets(self, _: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [dataset.to_dict() for dataset in self.catalog.datasets]

    def _list_models(self, _: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [model.to_dict() for model in self.catalog.models]

    @staticmethod
    def _plan_research(arguments: Mapping[str, Any]) -> dict[str, Any]:
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        return AgentOrchestrator().plan(prompt).to_dict()

    def _recommend_models(self, arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw_task = dict(arguments.get("task", {}))
        if not raw_task:
            raise ValueError("task is required")
        # JSON arrays are normalized to the tuple fields used by TaskSpec.
        for key in ("physics",):
            if key in raw_task:
                raw_task[key] = tuple(raw_task[key])
        task = TaskSpec(**raw_task)
        rows = recommend_models(
            task,
            self.catalog.models,
            top_k=int(arguments.get("top_k", 8)),
            evidence_weight=float(arguments.get("evidence_weight", 0.70)),
        )
        return [row.to_dict() if hasattr(row, "to_dict") else asdict(row) for row in rows]

    @staticmethod
    def _list_metric_suites(_: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "suites": {name: list(members) for name, members in SUITES.items()},
            "metrics": sorted(METRICS),
        }

    @staticmethod
    def _audit_figure_spec(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return audit_figure_spec(FigureSpec.from_dict(arguments["spec"])).to_dict()
