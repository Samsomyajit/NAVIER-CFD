
export const ALGORITHM_VERSION = "0.1.0";

export function normalizeToken(value) {
  return String(value ?? "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
}

export function normalizeTokens(values) {
  return new Set((values ?? []).map(normalizeToken));
}

export function expandPhysics(values) {
  const out = normalizeTokens(values);
  const fluidAliases = new Set([
    "fluid_dynamics","cfd","navier_stokes","incompressible_navier_stokes",
    "compressible_navier_stokes","aerodynamics","turbulence","fsi","combustion",
    "scalar_transport","free_surface"
  ]);
  const particleAliases = new Set(["particle","particles","granular","multiphase","dem","sph"]);
  if ([...out].some(x => fluidAliases.has(x))) {
    out.add("fluid_dynamics");
    out.add("general_pde");
  }
  if ([...out].some(x => particleAliases.has(x))) {
    out.add("particle");
    out.add("granular");
  }
  return out;
}

function matches(value, supported) {
  const normalized = normalizeTokens(supported);
  return normalized.has("any") || normalized.has(normalizeToken(value));
}

export function recommendModels(task, models, {topK=10, includeIncompatible=false}={}) {
  const recommendations = [];
  const taskPhysics = expandPhysics(task.physics ?? []);

  for (const model of models) {
    let score = 0;
    const reasons = [];
    const cautions = [];
    let compatible = true;

    if ((model.dimensions ?? []).includes(Number(task.dimension))) {
      score += 14; reasons.push(`supports ${task.dimension}D`);
    } else {
      compatible = false; cautions.push(`does not list ${task.dimension}D support`);
    }

    if (matches(task.mesh_type, model.mesh_types)) {
      score += 14; reasons.push(`compatible with ${task.mesh_type} meshes`);
    } else {
      compatible = false; cautions.push(`mesh mismatch: ${task.mesh_type}`);
    }

    if (matches(task.temporal_mode, model.temporal_modes)) {
      score += 10; reasons.push(`supports ${task.temporal_mode} operation`);
    } else {
      cautions.push(`temporal mode not demonstrated: ${task.temporal_mode}`);
    }

    if (matches(task.geometry_mode, model.geometry_modes)) {
      score += 12; reasons.push(`supports ${task.geometry_mode} geometry`);
    } else if (task.requires_geometry_transfer) {
      compatible = false; cautions.push("geometry transfer is required but unsupported");
    }

    const modelTasks = normalizeTokens(model.tasks);
    const taskType = normalizeToken(task.task_type);
    if (modelTasks.has(taskType) || modelTasks.has("general")) {
      score += 20; reasons.push(`matches task type ${task.task_type}`);
    } else if (normalizeTokens(model.categories).has(taskType)) {
      score += 15; reasons.push(`category matches ${task.task_type}`);
    } else {
      cautions.push(`task type ${task.task_type} is not a primary use case`);
    }

    const modelPhysics = expandPhysics(model.physics ?? []);
    const overlap = [...taskPhysics].filter(x => modelPhysics.has(x));
    if (overlap.length) {
      const bonus = Math.min(12, 4 * overlap.length);
      score += bonus; reasons.push(`physics overlap: ${overlap.sort().join(", ")}`);
    } else if (taskPhysics.size && !modelPhysics.has("general_pde")) {
      cautions.push("no explicit physics-family match");
    }

    const categories = new Set(model.categories ?? []);
    const tags = new Set(model.tags ?? []);

    if (task.requires_conservation) {
      if (categories.has("acceleration") || categories.has("physics_informed") || tags.has("conservative")) {
        score += 8; reasons.push("has a physics/solver pathway for conservation");
      } else cautions.push("conservation is not guaranteed by the model card");
    }

    if (task.requires_uncertainty) {
      if (categories.has("uncertainty") || tags.has("uncertainty")) {
        score += 8; reasons.push("uncertainty support is documented");
      } else cautions.push("requires an external UQ wrapper");
    }

    if (task.requires_long_rollout) {
      if (tags.has("long_rollout") || categories.has("acceleration")) {
        score += 8; reasons.push("designed or evaluated for long rollouts");
      } else cautions.push("long-horizon stability is not a core feature");
    }

    if (task.requires_mesh_transfer) {
      if (tags.has("mesh_transfer") || categories.has("geometry")) {
        score += 8; reasons.push("mesh/geometry transfer is part of the architecture");
      } else {
        compatible = false; cautions.push("mesh transfer is required but not established");
      }
    }

    if (task.hardware_memory_gb != null && model.min_memory_gb != null) {
      if (Number(task.hardware_memory_gb) >= Number(model.min_memory_gb)) {
        score += 4; reasons.push("fits the declared memory budget");
      } else {
        compatible = false;
        cautions.push(`estimated minimum memory ${model.min_memory_gb} GB exceeds budget`);
      }
    }

    if (task.preferred_framework && model.framework &&
        normalizeToken(task.preferred_framework) === normalizeToken(model.framework)) {
      score += 3; reasons.push(`uses preferred framework ${model.framework}`);
    }

    score += ({native:3, adapter:2, external:1, metadata:0})[model.integration] ?? 0;

    if (compatible || includeIncompatible) {
      recommendations.push({model, score, compatible, reasons, cautions});
    }
  }

  recommendations.sort((a,b) => (b.score-a.score) || a.model.name.localeCompare(b.model.name));
  return recommendations.slice(0, topK);
}

export function recommendDatasets(task, datasets, {topK=6}={}) {
  const scored = [];
  const physics = expandPhysics(task.physics ?? []);
  for (const dataset of datasets) {
    let score = 0;
    const reasons = [];
    const cautions = [];
    if ((dataset.dimensions ?? []).includes(Number(task.dimension))) {
      score += 18; reasons.push(`contains ${task.dimension}D cases`);
    } else cautions.push(`no declared ${task.dimension}D cases`);

    if (matches(task.mesh_type, dataset.mesh_types)) {
      score += 18; reasons.push(`matches ${task.mesh_type} representation`);
    } else cautions.push(`mesh representation differs from ${task.mesh_type}`);

    if (matches(task.temporal_mode, dataset.temporal_modes)) {
      score += 14; reasons.push(`supports ${task.temporal_mode} tasks`);
    } else cautions.push(`temporal mode differs from ${task.temporal_mode}`);

    if (matches(task.geometry_mode, dataset.geometry_modes)) {
      score += 14; reasons.push(`supports ${task.geometry_mode} geometry`);
    } else cautions.push(`geometry mode differs from ${task.geometry_mode}`);

    const datasetPhysics = expandPhysics(dataset.physics ?? []);
    const overlap = [...physics].filter(x => datasetPhysics.has(x));
    if (overlap.length) {
      score += Math.min(16, overlap.length * 5);
      reasons.push(`physics overlap: ${overlap.sort().join(", ")}`);
    }

    const scenarioText = (dataset.scenarios ?? []).map(normalizeToken).join(" ");
    const problemTokens = normalizeToken(task.problem).split("_").filter(x => x.length > 2);
    const scenarioHits = problemTokens.filter(x => scenarioText.includes(x));
    if (scenarioHits.length) {
      score += Math.min(16, scenarioHits.length * 8);
      reasons.push(`target-case overlap: ${scenarioHits.join(", ")}`);
    }

    if (dataset.hf_repo_id) {
      score += 4; reasons.push("direct Hugging Face integration is available");
    } else cautions.push("requires external dataset acquisition");

    scored.push({dataset, score, reasons, cautions});
  }
  scored.sort((a,b) => (b.score-a.score) || a.dataset.name.localeCompare(b.dataset.name));
  return scored.slice(0, topK);
}

export function buildManifest(task, modelResults, datasetResults) {
  return {
    schema: "navier-cfd.run-manifest/v1",
    recommender_version: ALGORITHM_VERSION,
    generated_at: new Date().toISOString(),
    task,
    recommended_models: modelResults.map((r, rank) => ({
      rank: rank + 1, id: r.model.id, name: r.model.name, score: r.score,
      reasons: r.reasons, cautions: r.cautions, reference: r.model.reference
    })),
    recommended_datasets: datasetResults.map((r, rank) => ({
      rank: rank + 1, id: r.dataset.id, name: r.dataset.name, score: r.score,
      hf_repo_id: r.dataset.hf_repo_id, reasons: r.reasons, cautions: r.cautions
    })),
    validation_note: "Recommendations are architecture-compatibility hypotheses and require benchmark validation."
  };
}
