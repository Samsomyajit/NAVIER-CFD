from navier_cfd import Catalog, TaskSpec, recommend_models


def test_geometry_task_prefers_geometry_models():
    catalog = Catalog.load_builtin()
    task = TaskSpec(
        problem="3d_vehicle_aerodynamics",
        task_type="surrogate",
        dimension=3,
        mesh_type="point_cloud",
        temporal_mode="steady",
        geometry_mode="varying",
        physics=("aerodynamics",),
        requires_geometry_transfer=True,
        requires_mesh_transfer=True,
        hardware_memory_gb=80,
    )
    results = recommend_models(task, catalog.models, top_k=10)
    ids = {r.model.id for r in results}
    assert {"gino", "transolver", "upt"} & ids


def test_acceleration_task_selects_hybrid_models():
    catalog = Catalog.load_builtin()
    task = TaskSpec(
        problem="unsteady_cfd",
        task_type="acceleration",
        dimension=3,
        mesh_type="unstructured",
        temporal_mode="autoregressive",
        geometry_mode="varying",
        physics=("incompressible_navier_stokes",),
        requires_conservation=True,
        requires_long_rollout=True,
        hardware_memory_gb=80,
    )
    results = recommend_models(task, catalog.models, top_k=12)
    assert any(r.model.id in {"inc", "geometry_preconditioner", "np_newton"} for r in results)
