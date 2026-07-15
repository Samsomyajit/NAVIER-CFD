# AutoResearch architecture

NAVIER AutoResearch is an orchestration and governance layer around the existing NAVIER-CFD scientific stack. It does not replace datasets, models, training, metrics, or experiment manifests. It connects them to Codex through explicit skills and a controlled tool surface.

## System context

```mermaid
flowchart TB
    CLIENT[Client or researcher]
    CODEX[Codex]
    REPO[Repository instructions<br/>AGENTS.md]
    SKILLS[Workflow skills<br/>.agents/skills]
    MCP[Local STDIO MCP server]
    TOOLS[Deterministic ToolRegistry]
    SESSION[AutoResearchSession]
    CORE[NAVIER-CFD core]
    EXT[Approved external adapters]
    OUTPUT[Research package]

    CLIENT --> CODEX
    REPO --> CODEX
    SKILLS --> CODEX
    CODEX <--> MCP
    MCP --> TOOLS
    CODEX <--> SESSION
    TOOLS --> CORE
    SESSION --> CORE
    CORE --> OUTPUT
    SESSION --> OUTPUT
    CODEX -. approval-gated .-> EXT
    EXT --> CORE
```

## Architectural layers

| Layer | Components | Responsibility |
|---|---|---|
| Interaction | Client, Codex | Interpret goals, present evidence, request approval. |
| Instruction | `AGENTS.md`, `.agents/skills` | Define scientific rules and reusable workflows. |
| Tool | MCP server, `ToolRegistry` | Expose explicit deterministic operations. |
| Governance | `ResearchContract`, `AutoResearchSession` | Enforce permissions, approvals, budgets, state, and stopping. |
| Scientific core | providers, `CFDSample`, model hub, recommender, trainer, metrics | Perform calculations and preserve scientific contracts. |
| Analysis | diagnostics and FigureLab | Localize errors, render figures, audit integrity. |
| Evidence | JSON/JSONL workspaces, checkpoints, manifests | Preserve provenance, actions, findings, and decisions. |
| Execution adapters | future training, solver, Slurm, storage connectors | Perform approved expensive or external actions. |

## Component flow

```mermaid
flowchart LR
    P[Problem statement]
    OBJ[ResearchObjective]
    CON[ResearchContract]
    PLAN[AgentOrchestrator plan]
    AUDIT[Data audit]
    HYP[Hypotheses]
    CAMP[Campaign proposal]
    APP{Approval required?}
    EXEC[Explicit execution adapter]
    EVAL[Metrics and diagnostics]
    REPLAN{Stop or replan?}
    FIG[FigureLab]
    PACK[Evidence package]

    P --> OBJ --> CON --> PLAN
    PLAN --> AUDIT --> HYP --> CAMP --> APP
    APP -- no --> EXEC
    APP -- approved --> EXEC
    APP -- rejected --> CAMP
    EXEC --> EVAL --> REPLAN
    REPLAN -- replan --> HYP
    REPLAN -- stop --> FIG --> PACK
```

## Planning path

The existing `AgentOrchestrator` remains the planner.

```mermaid
flowchart LR
    NLP[Natural-language objective]
    INT[Deterministic task interpretation]
    TASK[TaskSpec]
    DSET[Dataset selection]
    COMP[Hard model compatibility]
    EVID[Paper-evidence scoring]
    RANK[Model shortlist]
    BENCH[Benchmark and ablation plan]
    MAN[Structured planner output]

    NLP --> INT --> TASK
    TASK --> DSET
    TASK --> COMP
    COMP --> EVID --> RANK
    DSET --> BENCH
    RANK --> BENCH --> MAN
```

An optional language model may improve extraction of the initial task fields, but deterministic schema validation and compatibility rules must remain authoritative.

## Tool-call path

```mermaid
sequenceDiagram
    participant C as Codex
    participant M as MCP server
    participant R as ToolRegistry
    participant N as NAVIER-CFD core

    C->>M: tools/list
    M->>R: specs()
    R-->>M: JSON schemas
    M-->>C: Available tools

    C->>M: tools/call(plan_research)
    M->>R: call(name, arguments)
    R->>N: AgentOrchestrator.plan()
    N-->>R: Structured plan
    R-->>M: JSON-safe result
    M-->>C: Tool result
```

The MCP server serializes results as JSON strings because the current tools return structured scientific metadata rather than binary data or interactive objects.

## Governance and approval path

```mermaid
flowchart TD
    PROPOSE[ActionProposal]
    ALLOW{Tool allowed?}
    RISK[Classify ActionRisk]
    APPROVE{Contract requires approval?}
    VOTE{Approved?}
    BUDGET{Within budget?}
    RUN[Execution adapter may run]
    RECORD[Record result and usage]
    DENY[Reject or revise action]

    PROPOSE --> ALLOW
    ALLOW -- no --> DENY
    ALLOW -- yes --> RISK --> APPROVE
    APPROVE -- no --> BUDGET
    APPROVE -- yes --> VOTE
    VOTE -- no --> DENY
    VOTE -- yes --> BUDGET
    BUDGET -- no --> DENY
    BUDGET -- yes --> RUN --> RECORD
```

### Action risk classes

| Risk | Examples | v1.1.0 expectation |
|---|---|---|
| `read` | list catalogues, inspect manifests, audit specifications | Normally automatic. |
| `write` | create configs, figures, reports, checkpoints | Approval by default. |
| `compute` | training, evaluation, large diagnostics | Approval by default. |
| `external` | downloads, remote APIs, cluster calls | Approval by default. |
| `destructive` | delete, overwrite, cancel permanent assets | Always tightly controlled. |

## Research memory

```mermaid
flowchart TB
    CONTRACT[research_contract.json]
    PLAN[research_plan.json]
    STATE[session_state.json]
    PROP[proposals.jsonl]
    APPR[approvals.jsonl]
    ACTION[actions.jsonl]
    FIND[findings.jsonl]
    DEC[decisions.jsonl]

    CONTRACT --> STATE
    PLAN --> STATE
    PROP --> APPR
    APPR --> ACTION
    ACTION --> FIND
    FIND --> DEC
    DEC --> STATE
```

### File roles

| File | Role |
|---|---|
| `research_contract.json` | Immutable campaign intent, budget, permissions, and stop policy. |
| `research_plan.json` | Planner output, recommended models, dataset, metrics, and benchmark design. |
| `session_state.json` | Current status, iteration counters, and cumulative resource usage. |
| `proposals.jsonl` | Proposed tool actions with reason, risk, arguments, and estimated cost. |
| `approvals.jsonl` | Human or policy approval decisions. |
| `actions.jsonl` | Planning events and action results. |
| `findings.jsonl` | Evidence-linked observations, calculations, interpretations, and hypotheses. |
| `decisions.jsonl` | Iteration-level stop or continue decisions. |

JSONL is used for append-only event histories. It preserves failed attempts and avoids silently rewriting earlier research decisions.

## AutoResearch state machine

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Planned: plan()
    Planned --> Ready: evidence and approvals prepared
    Ready --> Running: first action
    Running --> Paused: failure or missing approval
    Paused --> Running: recovery
    Running --> Running: next iteration
    Running --> Completed: objective satisfied
    Running --> Stopped: deterministic stop reason
    Running --> Failed: unrecoverable system failure
    Completed --> [*]
    Stopped --> [*]
    Failed --> [*]
```

`AutoResearchSession.evaluate_iteration()` currently changes a newly active campaign to `running` or `stopped`. Execution adapters may set `ready`, `paused`, `completed`, or `failed` as the workflow matures.

## Scientific claim boundary

```mermaid
flowchart LR
    OBS[Observed data]
    CALC[Deterministic calculation]
    INT[Interpretation]
    HYP[Hypothesis]
    CLAIM[Client-facing claim]

    OBS --> CALC --> INT --> HYP
    OBS --> CLAIM
    CALC --> CLAIM
    INT -. qualified .-> CLAIM
    HYP -. not yet established .-> CLAIM
```

AutoResearch records four kinds of statements separately:

1. **Observed facts** — values or metadata directly present in the source.
2. **Computed results** — outputs of deterministic tools.
3. **Interpretations** — domain reasoning supported by observations and calculations.
4. **Hypotheses** — falsifiable explanations requiring further experiments.

A hypothesis must not be presented as a verified finding.

## Security boundary

The v1.1.0 server is read-only. It does not expose:

- arbitrary shell execution;
- Python `eval` or dynamic code execution;
- automatic package installation;
- solver execution;
- Slurm submission;
- large downloads;
- credential access;
- file overwrite or deletion.

Official dataset providers separately enforce host allowlists, transfer limits, checksums, and safe archive handling.

## Extension pattern

Future tools should be added in this order:

1. define a deterministic Python function;
2. define its inputs and outputs;
3. create analytical and failure-case tests;
4. register a `ToolSpec`;
5. assign an `ActionRisk`;
6. integrate it with `ResearchContract`;
7. add approval and budget checks;
8. expose it through MCP;
9. document the tool and skill workflow;
10. add provenance to every result.

Never expose a high-risk command directly through MCP before the governance layer can account for it.
