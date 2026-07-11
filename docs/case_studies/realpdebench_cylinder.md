# RealPDEBench: cylinder sim-to-real forecasting

## Goal
Measure whether simulation pretraining improves forecasting on paired experimental cylinder-wake data.

## Data
```bash
navier datasets download realpdebench --local-dir ./data/realpdebench --pattern "cylinder/**"
```

## Training paradigms
1. numerical only;
2. real only;
3. numerical pretraining followed by real fine-tuning.

## Recommended models
PIBERT, FourierFlow, FNO, Transolver, UPT, DPOT, DeepONet, and DMD.

## Metrics
RMSE, MAE, relative L2, R², frequency-domain RMSE, kinetic-energy error, update ratio, and OOD parameter performance.

## References
Hu et al., *RealPDEBench*, ICLR 2026; Chakraborty, Pan & Chen, PIBERT, 2026.
