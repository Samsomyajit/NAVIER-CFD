---
name: navier-audit-cfd-data
description: Audit CFD, PDE, MFiX, DEM, mesh, particle, or experimental data before surrogate modelling. Use for onboarding files, checking dataset consistency, leakage, fields, units, masks, geometry, temporal windows, operating conditions, provenance, or readiness; do not train models before completing the audit.
---

# Audit CFD data

1. Inspect files without executing arbitrary pickle or untrusted code.
2. Identify representation: structured grid, unstructured mesh, point cloud, particle set, scalar table, sensor series, or hybrid.
3. Record shapes, coordinates, fields, phases, units, masks, time spacing, geometry, operating conditions, boundary conditions, solver, fidelity, and provenance.
4. Check finite values, constants, impossible ranges, duplicate cases, missing fields, inconsistent dimensions, and unit conflicts.
5. Detect leakage from overlapping temporal windows, repeated geometries, repeated seeds, or derived samples crossing splits.
6. Preserve official provider splits when available.
7. State which physical metrics are valid and which required metadata are missing.
8. Produce a readiness decision: ready, conditionally ready, or not ready.
9. Save the audit evidence and unresolved risks in the AutoResearch workspace.

Do not infer units or field meanings from file names alone when uncertainty remains.
