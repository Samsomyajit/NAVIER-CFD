from __future__ import annotations

from navier_cfd import Catalog, TaskSpec, load_builtin_evidence, recommend_models, score_model_evidence


def test_builtin_evidence_is_traceable_and_unique() -> None:
    records = load_builtin_evidence()
    assert len(records) >= 20
    assert len({record.id for record in records}) == len(records)
    assert all(record.source_url.startswith("https://") for record in records)
    assert all(record.paper_title and record.benchmark and record.metric_group for record in records)


def test_vehicle_task_uses_geometry_and_paper_evidence() -> None:
    task = TaskSpec(
        problem="vehicle_aerodynamics",
        task_type="surrogate",
        dimension=3,
        mesh_type="point_cloud",
        temporal_mode="steady",
        geometry_mode="varying",
        physics=("aerodynamics",),
        fidelity="rans",
        requires_geometry_transfer=True,
        requires_mesh_transfer=True,
        hardware_memory_gb=80,
    )
    results = recommend_models(task, Catalog.load_builtin().models, top_k=10)
    ids = [row.model.id for row in results]
    assert "gino" in ids[:6]
    assert "domino" in ids[:6]
    assert any(row.evidence_count > 0 for row in results[:3])
    assert all(0.0 <= row.score <= 100.0 for row in results)


def test_evidence_is_task_specific_not_global_popularity() -> None:
    records = load_builtin_evidence()
    cylinder = TaskSpec(
        problem="real_cylinder_wake",
        task_type="forecasting",
        dimension=2,
        mesh_type="structured",
        temporal_mode="autoregressive",
        geometry_mode="fixed",
        physics=("incompressible_navier_stokes",),
        fidelity="experiment",
        requires_long_rollout=True,
    )
    vehicle = TaskSpec(
        problem="vehicle_drag",
        task_type="surrogate",
        dimension=3,
        mesh_type="point_cloud",
        temporal_mode="steady",
        geometry_mode="varying",
        physics=("aerodynamics",),
        fidelity="rans",
        requires_geometry_transfer=True,
    )

    pibert_cylinder = score_model_evidence(cylinder, "pibert", records)
    pibert_vehicle = score_model_evidence(vehicle, "pibert", records)
    gino_vehicle = score_model_evidence(vehicle, "gino", records)

    assert pibert_cylinder.matched_count > 0
    assert pibert_cylinder.confidence > pibert_vehicle.confidence
    assert gino_vehicle.confidence > 0
    assert gino_vehicle.score > 0.5


def test_absolute_mse_without_baseline_is_neutral() -> None:
    from navier_cfd.evidence import EvidenceRecord, metric_utility

    record = EvidenceRecord(
        id="scale_ambiguous_mse",
        model_id="example",
        paper_title="Example",
        paper_year=2026,
        source_url="https://example.org/paper",
        benchmark="private_scaled_data",
        problem="flow",
        task_type="surrogate",
        metric_group="field_accuracy",
        metric="mse",
        value=1e-6,
        lower_is_better=True,
        physics=("fluid_dynamics",),
        dimension=2,
        mesh_type="structured",
        geometry_mode="fixed",
        temporal_mode="steady",
    )
    assert metric_utility(record) == 0.5
