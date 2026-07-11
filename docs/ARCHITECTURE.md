# Architecture

```mermaid
flowchart LR
  U[User / Agent / CLI] --> T[Typed TaskSpec]
  T --> R[Compatibility Filter + Explainable Recommender]
  C[(55-model Catalog)] --> R
  D[(11-dataset Catalog)] --> H[Hugging Face Manager]
  R --> P[Benchmark Planner]
  H --> P
  P --> X[Runner Adapters]
  X --> M[CFD Metrics + QoIs + Stability]
  M --> V[Run Manifest, Provenance, Leaderboard]
  A[Optional LLM Backend] --> U
```

## Layers

1. **Task layer:** physics, dimensionality, mesh, geometry, fidelity, temporal regime, hardware, and safety requirements.
2. **Catalog layer:** explicit model and dataset capabilities, references, and limitations.
3. **Data layer:** Hugging Face discovery, revision-pinned download, streaming, filtering, and caching.
4. **Recommendation layer:** hard compatibility filtering followed by explainable ranking.
5. **Execution layer:** native components and explicit adapters to official external implementations.
6. **Evaluation layer:** field, spectral, temporal, conservation, QoI, UQ, OOD, and cost metrics.
7. **Agent layer:** deterministic planning plus a provider-neutral LLM hook.

## Security boundary
External repositories are metadata-first. NAVIER-CFD does not clone, install, import, or execute third-party code automatically.
