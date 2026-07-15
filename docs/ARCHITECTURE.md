# NAVIER-CFD architecture



NAVIER-CFD is a layered platform. The model and dataset catalogues are not the platform by themselves; they are connected to canonical data adaptation, dataset-conditioned construction, training, physical evaluation, evidence-aware recommendation, and the AutoResearch governance layer.

## Platform flow

```mermaid
---
config:
  layout: elk
---
flowchart 
    U[User, Codex, or CLI]
    T[TaskSpec or ResearchContract]
    R[Compatibility and evidence recommender]
    C[(Model catalogue)]
    D[(Dataset catalogue)]
    P[Provider and canonical data layer]
    M[Dataset-conditioned model builder]
    X[Trainer and checkpoints]
    E[Metric suites and diagnostics]
    F[FigureLab]
    V[Manifests and research memory]

    U --> T --> R
    C --> R
    D --> P
    P --> M
    R --> M
    M --> X --> E --> F --> V
    T --> V
    P --> V
    R --> V
    X --> V
```

## Core layers

1. **Task layer** — physics, dimensionality, mesh, geometry, fidelity, temporal regime, hardware, generalization, and safety requirements.
2. **Catalogue layer** — explicit model and dataset capabilities, references, evidence, and limitations.
3. **Provider layer** — official-source probing, selective staging, authentication resolution, safe local loading, and access provenance.
4. **Canonical data layer** — `CFDSample`, `CFDBatch`, coordinates, masks, fields, parameters, and metadata.
5. **Construction layer** — dataset defaults, actual sample inference, user overrides, and model-specific translation.
6. **Recommendation layer** — hard compatibility filtering followed by task-matched paper-evidence scoring.
7. **Execution layer** — native model references, trainers, checkpoints, experiments, and explicit external adapters.
8. **Evaluation layer** — numerical, spectral, temporal, physical, profile, stability, and efficiency metrics.
9. **Analysis layer** — CFD diagnostics, error localization, and research-grade figures.
10. **Agent layer** — deterministic planning, Codex skills, MCP tools, contracts, approvals, budgets, and stopping rules.
11. **Evidence layer** — experiment manifests, upstream manifests, figure manifests, AutoResearch JSONL history, and hashes.

## Dataset-conditioned construction

```mermaid
flowchart BT
    DEFAULTS[Registered dataset defaults]
    SAMPLE[Actual CFDSample shapes and metadata]
    OVERRIDE[User overrides]
    KW[Explicit model keyword arguments]
    PLAN[Resolved ModelBuildPlan]
    MODEL[Executable model]

    DEFAULTS --> SAMPLE --> OVERRIDE --> KW --> PLAN --> MODEL
```

Higher layers override lower layers. The resolved build plan records the final interpretation.

## Dataset path

```mermaid
flowchart LR
    UP[Official upstream or authorized local data]
    PROBE[Provider probe]
    STAGE[Selective staging and checksum]
    LOAD[Provider-specific loader]
    SAMPLE[Canonical CFDSample]
    SPLIT[Official or declared split]
    TRAIN[Model and trainer]
    MAN[Access and experiment manifests]

    UP --> PROBE --> STAGE --> LOAD --> SAMPLE --> SPLIT --> TRAIN
    PROBE --> MAN
    STAGE --> MAN
    LOAD --> MAN
    SPLIT --> MAN
    TRAIN --> MAN
```

## Recommendation path

```mermaid
flowchart LR
    TASK[TaskSpec]
    HARD[Hard compatibility checks]
    PRIOR[Architecture prior]
    PAPERS[Registered paper evidence]
    MATCH[Task and metric similarity]
    CONF[Quality, coverage, and confidence]
    FINAL[Final score]
    LIST[Ranked shortlist with cautions]

    TASK --> HARD --> PRIOR --> FINAL
    TASK --> MATCH
    PAPERS --> MATCH --> CONF --> FINAL --> LIST
```

## AutoResearch path

```mermaid
flowchart LR
    CLIENT[Client problem]
    CODEX[Codex]
    SKILL[Repository skill]
    MCP[MCP tool]
    CONTRACT[ResearchContract]
    SESSION[AutoResearchSession]
    CORE[NAVIER scientific core]
    RESULT[Metrics, diagnostics, figures]
    MEMORY[Findings and decisions]

    CLIENT --> CODEX --> SKILL --> MCP --> CORE
    CODEX --> CONTRACT --> SESSION
    SESSION --> CORE --> RESULT --> MEMORY
    MEMORY --> CODEX
```

For the full agent architecture, see [AutoResearch architecture](AUTORESEARCH_ARCHITECTURE.md).

## Security boundaries

### External code

External repositories are metadata-first. NAVIER-CFD does not clone, install, import, or execute third-party code merely because a model is registered.

### Dataset access

Official-source staging uses:

- registered upstream hosts;
- transfer-size ceilings;
- SHA-256 manifests;
- checksum verification where published;
- safe archive extraction;
- no credential or license bypass.

### Agent tools

The v1.1.0 MCP surface is read-only. High-cost and destructive execution must use explicit approved adapters.

### Scientific validity

Unsupported physical metrics are marked invalid rather than fabricated. Units, axes, masks, and normalization remain explicit parts of the evaluation contract.

## Package map

```text
src/navier_cfd/
├── agents/          # deterministic task interpretation and planning
├── autoresearch/    # contracts, sessions, MCP server, tools, CLI
├── catalogs/        # model and dataset metadata
├── datasets/        # providers, canonical samples, splits, loading
├── diagnostics/     # deterministic CFD failure analysis
├── evidence/        # paper-level evidence records and scoring
├── figures/         # FigureSpec, audits, renderers, manifests
├── metrics/         # numerical, spectral, and physics suites
├── models/          # model hub and dataset-conditioned builders
├── training/        # common trainer and training results
├── checkpoints/     # portable checkpoints and manifests
└── experiment.py    # high-level end-to-end experiment API
```
