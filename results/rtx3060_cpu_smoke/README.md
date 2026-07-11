# CPU smoke-test evidence

This directory records an actual minimal execution of `experiments/rtx3060_demo.py` on a CPU-only environment. It verifies that dataset generation, model training, CFD metrics, recommendation, and run-manifest writing complete successfully.

Configuration: 16x16 grid, 8 training samples, 4 test samples, one TinyCNN epoch, seed 42. The persistence baseline wins because the Taylor-Green state changes only slightly over the short one-step horizon and the neural model is deliberately under-trained. This is an expected and useful behavior: the recommender should not prefer a neural model when a cheaper baseline is better.

These numbers are **not** publication benchmark results. Run the documented RTX 3060 configuration to produce the first GPU evidence table.
