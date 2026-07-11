from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_web_runtime_exports_catalog_and_runs_exact_recommender() -> None:
    subprocess.run([sys.executable, "scripts/build_web_runtime.py"], cwd=ROOT, check=True)

    catalog_path = ROOT / "website" / "data" / "catalog.json"
    runtime_path = ROOT / "website" / "runtime" / "navier_runtime.zip"
    assert catalog_path.exists()
    assert runtime_path.exists()

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert len(catalog["models"]) >= 50
    assert len(catalog["datasets"]) >= 8

    task = {
        "problem": "cylinder_wake",
        "task_type": "surrogate",
        "dimension": 2,
        "mesh_type": "structured",
        "temporal_mode": "autoregressive",
        "geometry_mode": "fixed",
        "physics": ["incompressible_flow"],
        "requires_conservation": True,
        "requires_long_rollout": True,
        "hardware_memory_gb": 24,
    }
    code = f"""
import json, sys
sys.path.insert(0, {str(runtime_path)!r})
from navier_web_bridge import health_json, recommend_json
health = json.loads(health_json())
assert health['status'] == 'ready'
assert health['models'] >= 50
ranked = json.loads(recommend_json({json.dumps(json.dumps(task))}, 5))
assert 1 <= len(ranked) <= 5
assert all('score' in item for item in ranked)
print(json.dumps({{'health': health, 'count': len(ranked)}}))
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_browser_javascript_has_valid_syntax() -> None:
    subprocess.run(["node", "--check", "website/app.js"], cwd=ROOT, check=True)
