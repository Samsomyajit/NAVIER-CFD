# Executable model hub

NAVIER-CFD 0.3 provides a uniform Python interface for discovering, constructing, and connecting neural CFD models.

## Installation

The catalog, recommender, evidence database, datasets, and planning tools are available in the core package:

```bash
pip install navier-cfd
```

Native executable models require PyTorch:

```bash
pip install "navier-cfd[models]"
```

## Model states

Every catalog entry has a `ModelHandle` and one of four runtime states:

| State | Meaning |
|---|---|
| `native` | NAVIER-CFD ships an executable implementation. |
| `external_adapter` | A stable import entrypoint for an installed upstream package is registered. |
| `source_available` | An official source repository is known, but no stable constructor is registered. |
| `metadata` | The scientific model card exists, but an executable adapter is still required. |

Inspect the full registry:

```python
from navier_cfd import list_models

for handle in list_models():
    status = handle.status
    print(handle.id, status.mode, status.executable, status.dependency_available)
```

## Native models

The first native runtime release includes PINN, DeepONet, and FNO reference implementations.

### FNO

```python
from navier_cfd import TaskSpec, load_model

flow = TaskSpec(
    problem="cylinder_wake",
    task_type="forecasting",
    dimension=2,
    mesh_type="structured",
    temporal_mode="autoregressive",
    geometry_mode="fixed",
    physics=("incompressible_navier_stokes",),
)

model = load_model(
    "fno",
    task=flow,
    in_channels=3,
    out_channels=2,
    modes=(16, 16),
    width=64,
    n_layers=4,
)
```

The native FNO supports 1D, 2D, and 3D tensors. Inputs are channel-last by default:

```text
1D: [batch, x, channels]
2D: [batch, x, y, channels]
3D: [batch, x, y, z, channels]
```

Set `channel_last=False` for standard PyTorch channel-first tensors.

### PINN

```python
pinn = load_model(
    "pinn",
    task=flow,
    input_dim=3,
    output_dim=3,
    hidden_channels=128,
    depth=5,
    activation="tanh",
)
```

The PINN builder supplies the coordinate network. Governing-equation residuals, boundary conditions, initial conditions, nondimensionalization, and loss balancing remain explicit user code because they depend on the selected PDE and discretization.

### DeepONet

```python
deeponet = load_model(
    "deeponet",
    task=flow,
    branch_input_dim=256,
    trunk_input_dim=2,
    output_dim=3,
    latent_dim=128,
)
```

The returned module accepts a branch tensor `[B, branch_input_dim]` and a trunk tensor `[N, trunk_input_dim]` or `[B, N, trunk_input_dim]`.

## Connecting upstream implementations

NAVIER-CFD does not silently clone or execute research repositories. This avoids dependency conflicts, unreviewed setup code, and accidental license violations.

An installed package can be connected using a stable import entrypoint:

```python
from navier_cfd import ModelHub

hub = ModelHub()
hub.register_external(
    "transolver",
    entrypoint="my_transolver_package:Transolver",
    install_spec="my-transolver-package",
    repository="https://github.com/thuml/Neural-Solver-Library",
)

model = hub.load("transolver", hidden_dim=256, num_layers=8)
```

An entrypoint can also be supplied for one call:

```python
model = hub.load(
    "transolver",
    entrypoint="my_transolver_package:Transolver",
    hidden_dim=256,
)
```

## Opt-in dependency installation

Installation is never triggered by importing NAVIER-CFD. It requires explicit consent:

```python
hub.model("transolver").install(allow_external=True)
```

The installer invokes `python -m pip install` without a shell. Review the package source, upstream license, model weights, and data license before enabling installation.

## Registering a project-local builder

A research project can register a constructor directly:

```python
from navier_cfd import ModelHub

hub = ModelHub()

def build_my_model(*, task, spec, width=128, **kwargs):
    return MyModel(dimension=task.dimension, width=width, **kwargs)

hub.register_builder("gino", build_my_model)
model = hub.load("gino", task=flow, width=256)
```

This makes the project implementation available through the same model identifier used by the recommender and evidence catalog.

## Scientific interpretation

A common loader does not make architectures numerically interchangeable. Each model still requires its own:

- input/output variable definition;
- normalization and nondimensionalization;
- grid, mesh, point-cloud, or particle representation;
- boundary-condition encoding;
- temporal stepping strategy;
- training loss and optimization setup;
- official weights and license checks;
- benchmark validation.

The model hub standardizes discovery, construction, status reporting, and adapter registration. It does not erase the scientific assumptions of the original method.
