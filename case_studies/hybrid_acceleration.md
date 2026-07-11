# Hybrid CFD acceleration

## Goal
Accelerate a conventional solver without losing residual checks, conservation, or fallback.

## Candidate methods
INC, Solver-in-the-Loop, PICT, NeuroSEM, neural-operator preconditioned Newton, and geometry-aware neural preconditioners.

## Experimental protocol
- compare against the unaccelerated solver and classical preconditioners;
- use matched-error wall-clock and iteration counts;
- test unseen meshes, timesteps, and solver settings;
- report accepted learned steps, rejected steps, fallback cost, residual reduction, and conservation defects;
- include long-rollout turbulence statistics for unsteady cases.

## Agent command
```bash
navier agent plan "Accelerate unsteady 3D incompressible CFD on unstructured meshes with conservation, uncertainty, and fallback"
```

## References
Solver-in-the-Loop; INC; PICT; NeuroSEM; neural operator preconditioning.
