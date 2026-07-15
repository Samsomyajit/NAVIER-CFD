---
name: navier-generate-research-figures
description: Design, render, and audit publication-grade CFD and surrogate figures with traceable FigureSpec and FigureManifest files. Use for truth-prediction-error fields, profiles, rollouts, spectra, parity, ablations, uncertainty, and interface diagnostics; do not create decorative plots detached from the scientific question.
---

# Generate research-grade NAVIER figures

1. Identify the scientific question each figure must answer.
2. Create a `FigureSpec` before rendering.
3. Use shared truth/prediction color limits and a separately defined error scale.
4. Declare units, masks, normalization, interpolation, cases, times, field mapping, and selection rule.
5. Prefer representative, worst-case, median, or pre-registered selections over favorable examples.
6. Use physical aspect ratios and coordinates; do not plot non-uniform grids as uniform images without correct geometry.
7. Export PDF or SVG plus a high-resolution PNG.
8. Run the figure audit and fix all errors.
9. Save a `FigureManifest` with source run, dataset/checkpoint hashes, commit, specification, and outputs.
10. Write a factual caption that separates observation from interpretation.

Never label normalized values with physical units.
