# RTX 3060 quick start: first end-to-end NAVIER-CFD result

This workflow is a deliberately small, reproducible proof that the NAVIER-CFD pipeline works before downloading multi-gigabyte benchmark datasets. It generates an analytical two-dimensional Taylor-Green vortex dataset, trains compact CNN and Fourier neural-operator baselines, computes CFD-aware metrics, and writes an evidence-based recommendation and complete run manifest.

The demo is **not** a publication benchmark. Its purpose is to validate installation, CUDA, mixed precision, model training, metric calculation, recommendation, and provenance capture on a 12 GB RTX 3060.

## 1. Recommended machine

- NVIDIA RTX 3060, 12 GB VRAM
- 32 GB system RAM minimum; 64 GB recommended for later CFD datasets
- 1-2 TB NVMe SSD
- Ubuntu 22.04/24.04 or Windows 11 with WSL2
- Recent NVIDIA driver

## 2. Transfer and install

```bash
git clone https://github.com/Samsomyajit/NAVIER-CFD.git
cd NAVIER-CFD
git checkout rtx3060-mvp-benchmark

conda env create -f environment-rtx3060.yml
conda activate navier-cfd

# CUDA 12.1 example. Use the PyTorch selector if your driver requires another build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Check the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Fast smoke test

```bash
python experiments/rtx3060_demo.py \
  --grid 32 \
  --train-samples 64 \
  --test-samples 16 \
  --epochs 2 \
  --batch-size 8 \
  --models tinycnn,tinyfno \
  --output-dir results/rtx3060_smoke
```

Expected outputs:

```text
results/rtx3060_smoke/
  metrics.csv
  recommendation.json
  run_manifest.json
  tinycnn.pt
  tinyfno.pt
```

## 4. Recommended first RTX 3060 run

```bash
python experiments/rtx3060_demo.py \
  --grid 64 \
  --train-samples 512 \
  --test-samples 128 \
  --epochs 20 \
  --batch-size 8 \
  --models tinycnn,tinyfno \
  --output-dir results/rtx3060_demo
```

If memory is tight, reduce `--batch-size` to 4 or 2. The script uses CUDA mixed precision automatically and runs models sequentially.

## 5. What the proof demonstrates

1. A versioned CFD task can be generated and reproduced from a seed.
2. Multiple neural solver families can be trained under one protocol.
3. The same field, spectral, divergence, latency, and parameter metrics are computed for every model.
4. A transparent task-aware score produces a ranked recommendation.
5. Every run records its configuration, device, software versions, metrics, and model checkpoints.

A persistence baseline is intentionally included. If it wins a very short one-step task, that is a valid result rather than a failure: the recommender should prefer the cheapest adequate method.

## 6. Next benchmark phase

After the demo passes, add datasets in this order:

1. CFDBench cavity and cylinder subsets
2. RealPDEBench cylinder and fluid-structure interaction
3. One PDEBench Navier-Stokes task

Then replace the compact models with the official or verified adapters for U-Net, FNO, DeepONet, PIBERT, and Transolver. FourierFlow belongs in a separate probabilistic track, and INC belongs in a solver-coupled correction track.

## 7. Reproducibility rule

Do not edit result JSON manually. Keep the generated `run_manifest.json`, exact Git commit, dataset revision, preprocessing settings, seed, and hardware description with every reported result.
