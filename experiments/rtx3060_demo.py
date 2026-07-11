#!/usr/bin/env python3
"""RTX-3060 friendly end-to-end NAVIER-CFD benchmark demonstration.

Generates an analytical 2-D Taylor-Green vortex dataset, trains compact CNN and
Fourier neural-operator baselines, computes CFD-aware metrics, and produces an
evidence-based recommendation plus a reproducible run manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    grid: int = 32
    train_samples: int = 256
    test_samples: int = 64
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    viscosity_min: float = 0.005
    viscosity_max: float = 0.05
    dt: float = 0.1
    seed: int = 42
    width: int = 20
    modes: int = 8
    output_dir: str = "results/rtx3060_demo"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(n: int, cfg: Config, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, cfg.grid, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    inputs, targets = [], []
    for _ in range(n):
        nu = rng.uniform(cfg.viscosity_min, cfg.viscosity_max)
        t0 = rng.uniform(0.0, 1.0)
        amp = rng.uniform(0.7, 1.3)
        px, py = rng.uniform(0.0, 2.0 * np.pi, size=2)

        def field(t: float) -> np.ndarray:
            decay = amp * np.exp(-2.0 * nu * t)
            u = decay * np.sin(xx + px) * np.cos(yy + py)
            v = -decay * np.cos(xx + px) * np.sin(yy + py)
            return np.stack([u, v], axis=0).astype(np.float32)

        state = field(t0)
        next_state = field(t0 + cfg.dt)
        nu_channel = np.full((1, cfg.grid, cfg.grid), nu, dtype=np.float32)
        dt_channel = np.full((1, cfg.grid, cfg.grid), cfg.dt, dtype=np.float32)
        inputs.append(np.concatenate([state, nu_channel, dt_channel], axis=0))
        targets.append(next_state)
    return torch.tensor(np.stack(inputs)), torch.tensor(np.stack(targets))


class TinyCNN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2] + self.net(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes, modes)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            b,
            self.weight_pos.shape[1],
            nx,
            ny // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes, nx)
        m2 = min(self.modes, ny // 2 + 1)
        out_ft[:, :, :m1, :m2] = self._mul(
            x_ft[:, :, :m1, :m2], self.weight_pos[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self._mul(
            x_ft[:, :, -m1:, :m2], self.weight_neg[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class TinyFNO(nn.Module):
    def __init__(self, width: int = 20, modes: int = 8) -> None:
        super().__init__()
        self.lift = nn.Conv2d(4, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(3)]
        )
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(3)])
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x[:, :2]
        z = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            z = torch.nn.functional.gelu(spectral(z) + local(z))
        return residual + self.project(z)


def divergence_tensor(field: torch.Tensor) -> torch.Tensor:
    u, v = field[:, 0], field[:, 1]
    dx = 2.0 * math.pi / field.shape[-2]
    du_dx = (torch.roll(u, -1, -2) - torch.roll(u, 1, -2)) / (2.0 * dx)
    dv_dy = (torch.roll(v, -1, -1) - torch.roll(v, 1, -1)) / (2.0 * dx)
    return torch.sqrt(torch.mean((du_dx + dv_dy) ** 2))


def divergence_rms(field: torch.Tensor) -> float:
    return float(divergence_tensor(field).item())


def spectral_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = torch.abs(torch.fft.rfft2(pred, dim=(-2, -1)))
    t = torch.abs(torch.fft.rfft2(target, dim=(-2, -1)))
    numerator = torch.linalg.vector_norm(p - t)
    denominator = torch.linalg.vector_norm(t).clamp_min(1e-12)
    return float((numerator / denominator).item())


def evaluate(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    err = pred - target
    mse = float(torch.mean(err**2).item())
    rel = float(
        (torch.linalg.vector_norm(err) / torch.linalg.vector_norm(target).clamp_min(1e-12)).item()
    )
    total = torch.sum((target - torch.mean(target)) ** 2)
    r2 = float((1.0 - torch.sum(err**2) / total.clamp_min(1e-12)).item())
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "relative_l2": rel,
        "r2": r2,
        "spectral_relative_error": spectral_error(pred, target),
        "divergence_rms": divergence_rms(pred),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> tuple[nn.Module, float]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                pred = model(xb)
                field_loss = torch.mean((pred - yb) ** 2)
                div_loss = divergence_tensor(pred)
                loss = field_loss + 1e-3 * div_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    return model, time.perf_counter() - start


@torch.no_grad()
def infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    predictions, targets = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for xb, yb in loader:
        predictions.append(model(xb.to(device)).cpu())
        targets.append(yb)
    elapsed = time.perf_counter() - start
    return torch.cat(predictions), torch.cat(targets), elapsed


def recommendation(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str | int]]:
    """Rank candidates with an auditable accuracy/physics/cost score."""

    def minmax(key: str, invert: bool = False) -> dict[str, float]:
        vals = np.array([float(r[key]) for r in rows])
        if np.allclose(vals.max(), vals.min()):
            scores = np.ones_like(vals)
        else:
            scores = (vals - vals.min()) / (vals.max() - vals.min())
        if invert:
            scores = 1.0 - scores
        return {str(r["model"]): float(s) for r, s in zip(rows, scores)}

    rel = minmax("relative_l2", True)
    spec = minmax("spectral_relative_error", True)
    div = minmax("divergence_rms", True)
    speed = minmax("inference_ms_per_sample", True)
    ranked: list[dict[str, float | str | int]] = []
    for row in rows:
        name = str(row["model"])
        score = 100.0 * (
            0.45 * rel[name] + 0.25 * spec[name] + 0.20 * div[name] + 0.10 * speed[name]
        )
        ranked.append(
            {
                "model": name,
                "score": round(score, 3),
                "evidence": "45% field + 25% spectrum + 20% divergence + 10% latency",
            }
        )
    ranked.sort(key=lambda x: (-float(x["score"]), str(x["model"])))
    for i, item in enumerate(ranked, 1):
        item["rank"] = i
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/rtx3060_demo")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--models",
        default="tinycnn,tinyfno",
        help="Comma-separated trainable models: tinycnn,tinyfno",
    )
    args = parser.parse_args()
    cfg = Config(
        grid=args.grid,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dt=args.dt,
        width=args.width,
        modes=args.modes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    seed_everything(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    x_train, y_train = make_dataset(cfg.train_samples, cfg, cfg.seed)
    x_test, y_test = make_dataset(cfg.test_samples, cfg, cfg.seed + 1)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=cfg.batch_size)

    rows: list[dict[str, float | str]] = []
    start = time.perf_counter()
    pred = x_test[:, :2].clone()
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "model": "Persistence",
            **evaluate(pred, y_test),
            "train_seconds": 0.0,
            "inference_ms_per_sample": 1000.0 * elapsed / cfg.test_samples,
            "parameters": 0,
        }
    )

    selected = {token.strip().lower() for token in args.models.split(",") if token.strip()}
    candidates: list[tuple[str, nn.Module]] = []
    if "tinycnn" in selected:
        candidates.append(("TinyCNN", TinyCNN(cfg.width)))
    if "tinyfno" in selected:
        candidates.append(("TinyFNO", TinyFNO(cfg.width, cfg.modes)))
    if not candidates:
        raise ValueError("--models must include tinycnn and/or tinyfno")

    for name, model in candidates:
        model, train_seconds = train_model(model, train_loader, device, cfg)
        pred, target, infer_seconds = infer(model, test_loader, device)
        torch.save(model.state_dict(), out / f"{name.lower()}.pt")
        rows.append(
            {
                "model": name,
                **evaluate(pred, target),
                "train_seconds": train_seconds,
                "inference_ms_per_sample": 1000.0 * infer_seconds / cfg.test_samples,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ranking = recommendation(rows)
    (out / "recommendation.json").write_text(
        json.dumps(ranking, indent=2), encoding="utf-8"
    )
    manifest = {
        "task": "2D Taylor-Green vortex one-step forecasting",
        "config": asdict(cfg),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "results": rows,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"device": str(device), "results": rows, "ranking": ranking}, indent=2))


if __name__ == "__main__":
    main()
