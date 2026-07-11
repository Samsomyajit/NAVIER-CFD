"use strict";

const state = { pyodide: null, exact: false, catalog: null, results: [] };
const $ = (selector) => document.querySelector(selector);
const form = $("#task-form");
const resultsNode = $("#results");
const summaryNode = $("#result-summary");
const downloadButton = $("#download-results");

function setEngineStatus(kind, title, detail) {
  const dot = $("#status-dot");
  dot.className = `status-dot ${kind}`;
  $("#engine-status").textContent = title;
  $("#engine-detail").textContent = detail;
}

async function initialiseEngine() {
  try {
    if (typeof loadPyodide !== "function") throw new Error("Pyodide loader was unavailable");
    const pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"
    });
    const response = await fetch("./runtime/navier_runtime.zip", { cache: "no-store" });
    if (!response.ok) throw new Error(`Runtime download failed: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    pyodide.unpackArchive(bytes, "zip");
    await pyodide.runPythonAsync(
      "import sys\n" +
      "sys.path.insert(0, '.')\n" +
      "from navier_web_bridge import recommend_json, health_json\n"
    );
    const health = JSON.parse(pyodide.runPython("health_json()"));
    state.pyodide = pyodide;
    state.exact = true;
    setEngineStatus(
      "ready",
      "Exact Python recommender ready",
      `${health.models} models and ${health.datasets} datasets loaded from navier_cfd.recommender.`
    );
  } catch (error) {
    console.warn("Exact engine unavailable; using metadata fallback", error);
    await loadFallbackCatalog();
    setEngineStatus(
      "fallback",
      "Metadata fallback ready",
      "The exact Python runtime could not load. Recommendations will use the transparent browser heuristic."
    );
  }
}

async function loadFallbackCatalog() {
  if (state.catalog) return state.catalog;
  const response = await fetch("./data/catalog.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Catalog download failed: ${response.status}`);
  state.catalog = await response.json();
  return state.catalog;
}

function taskFromForm() {
  const data = new FormData(form);
  const task = {};
  for (const [key, value] of data.entries()) {
    if (key === "top_k") continue;
    task[key] = value;
  }
  for (const key of [
    "requires_conservation",
    "requires_long_rollout",
    "requires_geometry_transfer",
    "requires_mesh_transfer",
    "requires_uncertainty"
  ]) task[key] = data.has(key);
  task.dimension = Number(task.dimension);
  task.hardware_memory_gb = Number(task.hardware_memory_gb);
  task.physics = [task.physics];
  return task;
}

function flatten(value) {
  if (value === null || value === undefined) return [];
  if (Array.isArray(value)) return value.flatMap(flatten);
  if (typeof value === "object") return Object.entries(value).flatMap(([key, item]) => [key, ...flatten(item)]);
  return [String(value)];
}

function textBag(model) {
  return flatten(model).join(" ").toLowerCase().replaceAll("-", "_");
}

function fallbackRecommendations(task, topK) {
  const models = state.catalog?.models || [];
  const wanted = [task.problem, task.task_type, task.mesh_type, task.temporal_mode, task.geometry_mode, ...(task.physics || [])]
    .filter(Boolean).map((item) => String(item).toLowerCase().replaceAll("-", "_"));
  return models.map((model) => {
    const bag = textBag(model);
    let score = 20;
    const reasons = [];
    const cautions = ["Fallback score: verify with the exact Python engine or CLI before selecting a model."];
    for (const token of wanted) {
      const pieces = token.split("_").filter((part) => part.length > 2);
      const matches = pieces.filter((piece) => bag.includes(piece)).length;
      if (matches) {
        score += Math.min(12, matches * 4);
        reasons.push(`Metadata matches ${token.replaceAll("_", " ")}.`);
      }
    }
    const featureChecks = [
      ["requires_conservation", ["conservation", "physics_informed", "residual"]],
      ["requires_long_rollout", ["rollout", "autoregressive", "temporal"]],
      ["requires_geometry_transfer", ["geometry", "point_cloud", "unstructured"]],
      ["requires_mesh_transfer", ["mesh_transfer", "resolution", "operator"]],
      ["requires_uncertainty", ["uncertainty", "conformal", "probabilistic"]]
    ];
    for (const [field, terms] of featureChecks) {
      if (!task[field]) continue;
      if (terms.some((term) => bag.includes(term))) {
        score += 10;
        reasons.push(`Supports ${field.replace("requires_", "").replaceAll("_", " ")}.`);
      } else {
        score -= 5;
        cautions.push(`No explicit ${field.replace("requires_", "").replaceAll("_", " ")} evidence in the model card.`);
      }
    }
    const memory = Number(task.hardware_memory_gb || 0);
    const modelMemory = Number(model.min_memory_gb || model.minimum_memory_gb || 0);
    if (modelMemory && memory && modelMemory > memory) {
      score -= 18;
      cautions.push(`Model card suggests at least ${modelMemory} GB memory.`);
    }
    return { model, score: Math.max(0, Math.min(100, score)), reasons, cautions, engine: "fallback" };
  }).sort((a, b) => b.score - a.score).slice(0, topK);
}

async function exactRecommendations(task, topK) {
  const json = JSON.stringify(task);
  state.pyodide.globals.set("task_json_from_js", json);
  state.pyodide.globals.set("top_k_from_js", topK);
  const output = await state.pyodide.runPythonAsync("recommend_json(task_json_from_js, top_k_from_js)");
  return JSON.parse(output);
}

function firstValue(object, keys, fallback = "") {
  for (const key of keys) if (object && object[key] !== undefined && object[key] !== null) return object[key];
  return fallback;
}

function normaliseResult(result) {
  const model = result.model || result.model_spec || result;
  const scoreRaw = Number(firstValue(result, ["score", "total_score", "rank_score"], 0));
  const score = scoreRaw <= 1 && scoreRaw > 0 ? scoreRaw * 100 : scoreRaw;
  return {
    model,
    name: firstValue(model, ["name", "display_name", "id", "slug"], "Unnamed model"),
    category: firstValue(model, ["category", "family", "model_family", "numerical_role"], "model"),
    score: Math.max(0, Math.min(100, score)),
    reasons: firstValue(result, ["reasons", "strengths", "explanations"], []),
    cautions: firstValue(result, ["cautions", "limitations", "warnings"], []),
    summary: firstValue(model, ["summary", "description", "overview"], ""),
    tags: firstValue(model, ["tags", "tasks", "physics", "architectures"], [])
  };
}

function safeArray(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function renderResults(rawResults, task) {
  const results = rawResults.map(normaliseResult);
  state.results = { task, engine: state.exact ? "exact-python" : "fallback-js", results: rawResults };
  resultsNode.innerHTML = "";
  summaryNode.textContent = `${results.length} models ranked with the ${state.exact ? "exact Python" : "metadata fallback"} engine.`;
  if (!results.length) {
    resultsNode.innerHTML = '<div class="loading">No compatible models were returned. Relax one transfer or hardware constraint and try again.</div>';
    downloadButton.disabled = false;
    return;
  }
  results.forEach((result, index) => {
    const article = document.createElement("article");
    article.className = "result-card";
    const reasons = safeArray(result.reasons).slice(0, 6);
    const cautions = safeArray(result.cautions).slice(0, 5);
    const tags = safeArray(result.tags).flatMap((item) => typeof item === "string" ? [item] : []).slice(0, 8);
    article.innerHTML = `
      <span class="rank">${index + 1}</span>
      <p class="meta">${escapeHtml(String(result.category).replaceAll("_", " "))}</p>
      <h4>${escapeHtml(String(result.name))}</h4>
      ${result.summary ? `<p>${escapeHtml(String(result.summary))}</p>` : ""}
      <div class="score-row"><div class="score-track"><div class="score-fill" style="width:${result.score}%"></div></div><span class="score-value">${result.score.toFixed(1)}</span></div>
      ${tags.length ? `<div class="tag-list">${tags.map((tag) => `<span class="tag">${escapeHtml(tag.replaceAll("_", " "))}</span>`).join("")}</div>` : ""}
      <details open><summary>Why it was ranked</summary><ul class="reason">${(reasons.length ? reasons : ["Compatible with the selected task metadata."]).map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul></details>
      ${cautions.length ? `<details><summary>Cautions</summary><ul class="caution">${cautions.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul></details>` : ""}
    `;
    resultsNode.appendChild(article);
  });
  downloadButton.disabled = false;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[character]);
}

async function submitTask(event) {
  event.preventDefault();
  const task = taskFromForm();
  const topK = Number($("#top-k").value || 8);
  resultsNode.innerHTML = '<div class="loading">Ranking compatible model cards…</div>';
  summaryNode.textContent = "Running compatibility filters and evidence-aware scoring.";
  try {
    let ranked;
    if (state.exact) ranked = await exactRecommendations(task, topK);
    else {
      await loadFallbackCatalog();
      ranked = fallbackRecommendations(task, topK);
    }
    renderResults(ranked, task);
  } catch (error) {
    console.error(error);
    if (state.exact) {
      state.exact = false;
      await loadFallbackCatalog();
      setEngineStatus("fallback", "Metadata fallback active", "The Python engine rejected this task schema; a transparent fallback ranking was used.");
      renderResults(fallbackRecommendations(task, topK), task);
    } else {
      resultsNode.innerHTML = `<div class="loading">Recommendation failed: ${escapeHtml(error.message)}</div>`;
    }
  }
}

function applyPreset(preset) {
  const values = preset === "geometry" ? {
    problem: "3d_vehicle_aerodynamics", physics: "aerodynamics", task_type: "surrogate", dimension: "3", mesh_type: "point_cloud", geometry_mode: "varying", temporal_mode: "steady", hardware_memory_gb: "80"
  } : {
    problem: "unsteady_3d_cfd_acceleration", physics: "turbulence", task_type: "acceleration", dimension: "3", mesh_type: "unstructured", geometry_mode: "varying", temporal_mode: "autoregressive", hardware_memory_gb: "48"
  };
  for (const [name, value] of Object.entries(values)) form.elements[name].value = value;
  form.elements.requires_geometry_transfer.checked = true;
  form.elements.requires_mesh_transfer.checked = true;
  form.elements.requires_conservation.checked = true;
  form.elements.requires_long_rollout.checked = preset !== "geometry";
  form.scrollIntoView({ behavior: "smooth", block: "start" });
}

function downloadResults() {
  const blob = new Blob([JSON.stringify(state.results, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "navier-cfd-recommendations.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

form.addEventListener("submit", submitTask);
$("#preset-geometry").addEventListener("click", () => applyPreset("geometry"));
$("#preset-acceleration").addEventListener("click", () => applyPreset("acceleration"));
downloadButton.addEventListener("click", downloadResults);
window.addEventListener("DOMContentLoaded", initialiseEngine);
