import numpy as np

from navier_cfd.benchmarks.metrics import compute_metric_bundle


def test_metrics_identity():
    x = np.arange(16, dtype=float).reshape(4, 4)
    metrics = compute_metric_bundle(x, x)
    assert metrics["rmse"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert abs(metrics["r2"] - 1.0) < 1e-12
