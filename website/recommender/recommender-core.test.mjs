
import test from "node:test";
import assert from "node:assert/strict";
import {recommendModels, recommendDatasets} from "./recommender-core.mjs";

import {models, datasets} from "../data/catalog.mjs";

test("3D geometry task returns geometry-aware models", () => {
  const task = {
    problem:"3d_vehicle_aerodynamics", task_type:"surrogate", dimension:3,
    mesh_type:"point_cloud", temporal_mode:"steady", geometry_mode:"varying",
    physics:["aerodynamics"], requires_geometry_transfer:true, requires_mesh_transfer:true,
    requires_conservation:false, requires_uncertainty:false, requires_long_rollout:false,
    hardware_memory_gb:80, preferred_framework:"pytorch"
  };
  const ids = recommendModels(task, models, {topK:10}).map(x=>x.model.id);
  assert.ok(ids.some(x=>["gino","transolver","upt","domino","aerotransformer"].includes(x)));
});

test("hybrid acceleration task returns acceleration models", () => {
  const task = {
    problem:"unsteady_3d_cfd_acceleration", task_type:"acceleration", dimension:3,
    mesh_type:"unstructured", temporal_mode:"autoregressive", geometry_mode:"varying",
    physics:["incompressible_navier_stokes"], requires_geometry_transfer:true,
    requires_mesh_transfer:false, requires_conservation:true, requires_uncertainty:false,
    requires_long_rollout:true, hardware_memory_gb:80, preferred_framework:"pytorch"
  };
  const ids = recommendModels(task, models, {topK:12}).map(x=>x.model.id);
  assert.ok(ids.some(x=>["inc","geometry_preconditioner","np_newton","pict"].includes(x)));
});

test("dataset recommender surfaces CFD benchmarks", () => {
  const task = {
    problem:"cylinder_wake", task_type:"forecasting", dimension:2,
    mesh_type:"structured", temporal_mode:"autoregressive", geometry_mode:"fixed",
    physics:["incompressible_navier_stokes"]
  };
  const ids = recommendDatasets(task, datasets, {topK:5}).map(x=>x.dataset.id);
  assert.ok(ids.includes("cfdbench") || ids.includes("realpdebench") || ids.includes("pdebench"));
});
