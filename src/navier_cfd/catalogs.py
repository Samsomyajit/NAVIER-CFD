from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

from .specs import DatasetSpec, ModelSpec

_MODEL_ROWS = [
    ("pinn", "Physics-Informed Neural Network", ("physics_informed", "general_pde_solver", "inverse")),
    ("nsfnets", "NSFnets", ("physics_informed", "specialized")),
    ("pinnsformer", "PINNsFormer", ("physics_informed", "general_pde_solver")),
    ("deeponet", "DeepONet", ("surrogate", "general_pde_solver", "physics_informed")),
    ("mionet", "MIONet", ("surrogate", "general_pde_solver")),
    ("fourier_deeponet", "Fourier-DeepONet", ("surrogate", "specialized")),
    ("nested_fourier_deeponet", "Nested Fourier-DeepONet", ("surrogate", "specialized", "particle_multiphase")),
    ("fourier_mionet", "Fourier-MIONet", ("surrogate", "specialized", "particle_multiphase")),
    ("fno", "Fourier Neural Operator", ("surrogate", "general_pde_solver")),
    ("pino", "Physics-Informed Neural Operator", ("physics_informed", "surrogate", "general_pde_solver")),
    ("geo_fno", "Geo-FNO", ("geometry", "surrogate", "general_pde_solver")),
    ("gino", "Geometry-Informed Neural Operator", ("geometry", "surrogate")),
    ("u_fno", "U-FNO", ("surrogate", "specialized", "particle_multiphase")),
    ("f_fno", "Factorized FNO", ("surrogate", "general_pde_solver")),
    ("u_no", "U-shaped Neural Operator", ("surrogate", "general_pde_solver")),
    ("lsm", "Latent Spectral Model", ("surrogate", "general_pde_solver")),
    ("gnot", "General Neural Operator Transformer", ("surrogate", "general_pde_solver", "geometry")),
    ("galerkin_transformer", "Galerkin Transformer", ("surrogate", "general_pde_solver")),
    ("mwt", "Multiwavelet Transformer", ("surrogate", "general_pde_solver")),
    ("factformer", "FactFormer", ("surrogate", "general_pde_solver")),
    ("ono", "Orthogonal Neural Operator", ("surrogate", "general_pde_solver")),
    ("transolver", "Transolver", ("geometry", "surrogate", "general_pde_solver")),
    ("upt", "Universal Physics Transformer", ("foundation", "geometry", "surrogate", "general_pde_solver")),
    ("meshgraphnets", "MeshGraphNets", ("geometry", "surrogate")),
    ("domino", "DoMINO", ("geometry", "surrogate", "specialized")),
    ("pibert", "PIBERT", ("surrogate", "physics_informed", "specialized")),
    ("fourierflow", "FourierFlow", ("surrogate", "generative", "specialized")),
    ("pde_refiner", "PDE-Refiner", ("surrogate", "generative", "general_pde_solver")),
    ("dpot", "DPOT", ("foundation", "surrogate", "general_pde_solver")),
    ("poseidon", "Poseidon", ("foundation", "surrogate", "general_pde_solver")),
    ("prose_fd", "PROSE-FD", ("foundation", "surrogate", "specialized")),
    ("bcat", "BCAT", ("foundation", "surrogate", "specialized")),
    ("pdeformer1", "PDEformer-1", ("foundation", "general_pde_solver")),
    ("pi_mfm", "PI-MFM", ("foundation", "physics_informed", "general_pde_solver")),
    ("laplace_no", "Laplace Neural Operator", ("surrogate", "general_pde_solver")),
    ("state_space_no", "State-Space Neural Operator", ("surrogate", "general_pde_solver")),
    ("p3d", "P3D", ("surrogate", "foundation", "specialized")),
    ("aerotransformer", "AeroTransformer", ("foundation", "geometry", "specialized")),
    ("tadpole", "Tadpole", ("foundation", "surrogate")),
    ("solver_in_loop", "Solver-in-the-Loop", ("acceleration", "physics_informed")),
    ("inc", "Indirect Neural Corrector", ("acceleration", "physics_informed")),
    ("pict", "PICT", ("acceleration", "specialized", "physics_informed")),
    ("diffsph", "diffSPH", ("acceleration", "particle_multiphase", "specialized")),
    ("neurosem", "NeuroSEM", ("acceleration", "physics_informed", "specialized")),
    ("np_newton", "Neural-Operator Preconditioned Newton", ("acceleration", "general_pde_solver")),
    ("geometry_preconditioner", "Geometry-Aware Neural Preconditioner", ("acceleration", "geometry")),
    ("neuraldem", "NeuralDEM", ("specialized", "particle_multiphase", "surrogate")),
    ("revit", "ReViT", ("geometry", "surrogate")),
    ("deepmmnet", "DeepM&Mnet", ("surrogate", "specialized", "physics_informed")),
    ("conformal_deeponet", "Conformalized-DeepONet", ("uncertainty", "surrogate")),
    ("tante", "TANTE", ("surrogate", "general_pde_solver")),
    ("riemannonet", "RiemannONet", ("physics_informed", "specialized", "surrogate")),
    ("energy_transformer", "Energy Transformer Flow Reconstruction", ("inverse", "surrogate", "specialized")),
    ("fun_diff", "FunDiff", ("generative", "physics_informed", "general_pde_solver")),
    ("flow_matching_pde", "Flow Matching for PDEs", ("generative", "physics_informed", "general_pde_solver")),
]

_GEOMETRY = {"geo_fno", "gino", "gnot", "transolver", "upt", "meshgraphnets", "domino", "aerotransformer", "geometry_preconditioner", "revit"}
_THREED = {"gino", "upt", "domino", "p3d", "aerotransformer", "tadpole", "inc", "pict", "diffsph", "neurosem", "np_newton", "geometry_preconditioner", "neuraldem", "revit", "deepmmnet", "tante", "energy_transformer"}
_ACCEL = {"solver_in_loop", "inc", "pict", "diffsph", "neurosem", "np_newton", "geometry_preconditioner"}
_LONG = {"pinnsformer", "fno", "f_fno", "pibert", "fourierflow", "pde_refiner", "dpot", "poseidon", "prose_fd", "bcat", "p3d", "tadpole", "solver_in_loop", "inc", "neuraldem", "tante"}
_PARTICLE = {"diffsph", "neuraldem"}
_UQ = {"conformal_deeponet", "fun_diff", "fourierflow"}

_ARCH = {
    "pinn": "coordinate network with PDE residuals",
    "deeponet": "branch-trunk operator network",
    "fno": "Fourier neural operator",
    "pibert": "bidirectional transformer with Fourier-wavelet embeddings and physics-biased attention",
    "fourierflow": "frequency-guided generative flow model",
    "transolver": "physics-attention transformer on general geometries",
    "upt": "latent universal physics transformer",
    "gino": "graph lifting, latent FNO, and point projection",
    "domino": "decomposable multiscale iterative neural operator",
    "p3d": "scalable 3D global-context surrogate",
    "aerotransformer": "pretrained transformer for 3D aerodynamics",
    "inc": "equation-level indirect neural corrector",
    "pict": "differentiable GPU multi-block PISO solver",
    "diffsph": "differentiable smoothed-particle hydrodynamics",
    "np_newton": "neural-operator preconditioned Newton solver",
    "geometry_preconditioner": "geometry-aware neural iterative preconditioner",
}

_REF = {
    "pibert": "Chakraborty, Pan & Chen, 2026",
    "fno": "Li et al., ICLR 2021",
    "deeponet": "Lu et al., Nature Machine Intelligence 2021",
    "transolver": "Wu et al., ICML 2024",
    "upt": "Alkin et al., NeurIPS 2024",
    "gino": "Li et al., NeurIPS 2023",
    "inc": "Wei et al., NeurIPS 2025",
    "pict": "Franz et al., JCP 2025",
    "diffsph": "Winchenbach & Thuerey, JCP 2026",
}

_REPO = {
    "pibert": "https://github.com/Samsomyajit/pibert",
    "fno": "https://github.com/neuraloperator/neuraloperator",
    "geo_fno": "https://github.com/neuraloperator/Geo-FNO",
    "transolver": "https://github.com/thuml/Neural-Solver-Library",
    "inc": "https://github.com/tum-pbs/INC",
}


def _models() -> list[ModelSpec]:
    result: list[ModelSpec] = []
    for model_id, name, categories in _MODEL_ROWS:
        geometry = model_id in _GEOMETRY
        three_d = model_id in _THREED or model_id not in {"pinnsformer", "riemannonet"}
        accel = model_id in _ACCEL
        particle = model_id in _PARTICLE
        mesh_types = ("particle",) if particle else (("unstructured", "point_cloud", "structured") if geometry else ("structured", "meshfree"))
        result.append(ModelSpec(
            id=model_id,
            name=name,
            categories=categories,
            architecture=_ARCH.get(model_id, name + " architecture"),
            tasks=(("acceleration", "corrector", "preconditioner") if accel else ("surrogate", "forecasting")),
            physics=(("particle", "granular") if particle else ("general_pde", "fluid_dynamics")),
            mesh_types=mesh_types,
            geometry_modes=(("varying", "parameterized") if geometry else ("fixed", "parameterized")),
            temporal_modes=(("autoregressive", "unsteady", "steady") if model_id in _LONG or accel else ("steady", "unsteady")),
            dimensions=((1, 2, 3) if three_d else (1, 2)),
            strengths=("registered in the uniform NAVIER-CFD model taxonomy",),
            limitations=("verify claims on the selected CFD benchmark and discretization",),
            reference=_REF.get(model_id, name),
            repository=_REPO.get(model_id),
            integration="external" if model_id in _REPO else "metadata",
            min_memory_gb=24 if model_id in {"gino", "domino", "p3d", "aerotransformer", "neuraldem"} else 8,
            framework="pytorch",
            tags=tuple(filter(None, ["mesh_transfer" if geometry else "", "long_rollout" if model_id in _LONG or accel else "", "uncertainty" if model_id in _UQ else "", "conservative" if accel else "", "3d" if three_d else ""])),
        ))
    return result


def _datasets() -> list[DatasetSpec]:
    rows = [
        ("pdebench", "PDEBench", "Extensive time-dependent PDE benchmark", "AI4Science-WestlakeU/PDEBench", ("navier_stokes", "advection", "reaction_diffusion"), (1, 2, 3), ("structured",), ("fixed",), ("autoregressive",)),
        ("cfdbench", "CFDBench", "CFD benchmark with boundary, property, and geometry shifts", "chen-yingfa/CFDBench", ("cavity", "tube", "dam", "cylinder"), (2,), ("structured",), ("fixed", "varying"), ("steady", "autoregressive")),
        ("realpdebench", "RealPDEBench", "Paired real-world and simulated spatiotemporal systems", "AI4Science-WestlakeU/RealPDEBench", ("cylinder", "fsi", "controlled_cylinder", "foil", "combustion"), (2,), ("structured",), ("fixed", "varying"), ("autoregressive",)),
        ("airfrans", "AirfRANS", "RANS airfoil geometry and operating-condition benchmark", None, ("airfoil",), (2,), ("unstructured", "point_cloud"), ("varying",), ("steady",)),
        ("drivaernetpp", "DrivAerNet++", "Large multimodal vehicle CFD dataset", None, ("vehicle_aerodynamics",), (3,), ("point_cloud", "unstructured"), ("varying",), ("steady",)),
        ("drivaerml", "DrivAerML", "High-fidelity road-car aerodynamic CFD", None, ("vehicle_aerodynamics",), (3,), ("unstructured",), ("varying",), ("steady", "unsteady")),
        ("the_well", "The Well", "Diverse large-scale physics simulations", "polymathic-ai/the_well", ("fluid", "multiphysics"), (2, 3), ("structured",), ("fixed", "varying"), ("autoregressive",)),
        ("apebench", "APEBench", "Autoregressive PDE emulator benchmark", None, ("46_pde_configurations",), (1, 2, 3), ("structured",), ("fixed",), ("autoregressive",)),
        ("scalarflow", "ScalarFlow", "Real volumetric scalar transport flows", None, ("scalar_transport",), (3,), ("structured",), ("fixed",), ("autoregressive",)),
        ("shapenet_car", "ShapeNet-Car", "Vehicle-shape design benchmark", None, ("vehicle_design",), (3,), ("point_cloud",), ("varying",), ("steady",)),
        ("eagle", "EAGLE", "Fluid and geometry learning benchmark", None, ("fluid",), (2, 3), ("structured", "unstructured"), ("varying",), ("autoregressive",)),
    ]
    return [DatasetSpec(
        id=i, name=n, description=d, tasks=("surrogate", "benchmark"), physics=("fluid_dynamics", "general_pde"),
        dimensions=dim, mesh_types=mesh, geometry_modes=geo, temporal_modes=temp,
        hf_repo_id=hf, source_url=None, license=None, size=None, scenarios=scenarios,
        notes=("Pin dataset revisions and preserve official splits.",)
    ) for i, n, d, hf, scenarios, dim, mesh, geo, temp in rows]


@dataclass
class Catalog:
    models: list[ModelSpec]
    datasets: list[DatasetSpec]

    @classmethod
    def load_builtin(cls) -> "Catalog":
        return cls(models=_models(), datasets=_datasets())

    @classmethod
    def from_paths(cls, model_path: str | Path, dataset_path: str | Path) -> "Catalog":
        model_data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        dataset_data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        return cls([ModelSpec.from_dict(x) for x in model_data], [DatasetSpec.from_dict(x) for x in dataset_data])

    def model(self, model_id: str) -> ModelSpec:
        for item in self.models:
            if item.id == model_id:
                return item
        raise KeyError(f"Unknown model id: {model_id}")

    def dataset(self, dataset_id: str) -> DatasetSpec:
        for item in self.datasets:
            if item.id == dataset_id:
                return item
        raise KeyError(f"Unknown dataset id: {dataset_id}")

    def models_by_category(self, category: str) -> list[ModelSpec]:
        return [x for x in self.models if category in x.categories]

    def search_models(self, query: str) -> list[ModelSpec]:
        q = query.lower()
        return [x for x in self.models if q in x.name.lower() or q in x.architecture.lower() or any(q in t.lower() for t in x.tags)]

    def extend_models(self, models: Iterable[ModelSpec]) -> None:
        existing = {x.id for x in self.models}
        for model in models:
            if model.id in existing:
                raise ValueError(f"Duplicate model id: {model.id}")
            self.models.append(model)
            existing.add(model.id)
