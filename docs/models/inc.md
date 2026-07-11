# Indirect Neural Corrector

**Registry ID:** `inc`  
**Categories:** acceleration, physics-informed  
**Architecture:** equation-level corrector inserted into an autoregressive hybrid PDE solver rather than a direct state overwrite.

## Suitable tasks
Stable coarse-grid correction and long-horizon neural-numerical acceleration up to three-dimensional turbulence.

## Cautions
Requires integration with a numerical solver and matched-error wall-clock evaluation.

## Reference
Wei, Franz, List & Thuerey, *INC*, NeurIPS 2025. Code: https://github.com/tum-pbs/INC
