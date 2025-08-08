# NMF Performance Comparison (GPU-Accelerated)

PyTorch Device: mps
MLX Device: GPU (Metal)

| Framework | M x N | rank | Execution Time (s) | KL-Divergence |
|-----------|--- | ---|--------------------|---------------|
| PyTorch   | 1000 x 500 | 10 | 0.1711   | 25505.8242 |
| MLX       | 1000 x 500 | 10 | 0.0641   | 25505.8223   |
| PyTorch   | 200000 x 1000 | 10 | 21.2449   | 16945020.0000 |
| MLX       | 200000 x 1000 | 10 | 18.6933   | 16945018.0000   |
| PyTorch   | 100000 x 5000 | 20 | 66.2762   | 51499992.0000 |
| MLX       | 100000 x 5000 | 20 | 59.4625   | 51499988.0000   |
