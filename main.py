import time
import numpy as np
import torch
import mlx.core as mx
import scipy
import os
from pytorch_nmf import PyTorchNMF
from mlx_nmf import MLXNMF


# Ensure MLX uses GPU on Apple Silicon
try:
    mx.metal.set_cache_policy(mx.metal.CachePolicy.PREFER_SHARED)
except AttributeError:
    pass  # MLX will use CPU if not on Apple Silicon

# KL-divergence for PyTorch
def kl_div_torch(V, V_approx):
    V_approx = torch.clamp(V_approx, min=1e-10)
    return torch.sum(V * torch.log(V / V_approx) - V + V_approx).item()

# KL-divergence for MLX
def kl_div_mlx(V, V_approx):
    V = mx.array(V)
    V_approx = mx.maximum(V_approx, 1e-10)
    return float(mx.sum(V * mx.log(V / V_approx) - V + V_approx))

# Generate synthetic non-negative matrix
def generate_data(m=1000, n=500, r=20):
    np.random.seed(42)
    W_true = np.abs(np.random.randn(m, r))
    H_true = np.abs(np.random.randn(r, n))
    V = np.dot(W_true, H_true)

    return V

# Main comparison function
def compare_nmf(m, n, r, rank):
    # Generate data
    V_np = generate_data(m, n, r)
    m, n = V_np.shape
    W = np.abs(np.random.randn(m, rank))
    H = np.abs(np.random.randn(rank, n))


    device = torch.device("mps")
    V_torch = torch.from_numpy(V_np).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    H_torch = torch.from_numpy(H).float().to(device)
    V_mlx, W_mlx, H_mlx = mx.array(V_np), mx.array(W), mx.array(H)

    # PyTorch NMF
    start_time = time.perf_counter()
    torch_nmf = PyTorchNMF(W_torch, H_torch, V_torch.shape, rank=rank)
    torch_nmf.fit(V_torch)
    V_torch_approx = torch_nmf()
    torch_time = time.perf_counter() - start_time
    torch_kl = kl_div_torch(V_torch, V_torch_approx)

    # MLX NMF
    start_time = time.perf_counter()
    mlx_nmf = MLXNMF(W_mlx, H_mlx, V_mlx.shape, rank=rank)
    mlx_nmf.fit(V_mlx)
    V_mlx_approx = mlx_nmf()
    mlx_time = time.perf_counter() - start_time
    mlx_kl = kl_div_mlx(V_np, V_mlx_approx)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_report.md", "w") as f:
        f.write("# NMF Performance Comparison (GPU-Accelerated)\n\n")
        f.write(f"PyTorch Device: {device}\n")
        f.write(f"MLX Device: {'GPU (Metal)' if mx.metal.is_available() else 'CPU'}\n\n")
        f.write("| Framework | Execution Time (s) | KL-Divergence |\n")
        f.write("|-----------|--------------------|---------------|\n")
        f.write(f"| PyTorch   | {torch_time:.4f}   | {torch_kl:.4f} |\n")
        f.write(f"| MLX       | {mlx_time:.4f}   | {mlx_kl:.4f}   |\n")

    print(f"Results saved to results/comparison_report.md")
    print(f"PyTorch: Time={torch_time:.4f}s, KL-Divergence={torch_kl:.4f}")
    print(f"MLX: Time={mlx_time:.4f}s, KL-Divergence={mlx_kl:.4f}")

if __name__ == "__main__":
    
    print("Running NMF comparison...")
    compare_nmf(m=1000, n=500, r=20, rank=10)

    for i in range(5):
        print("Running with larger matrix sizes...")
        compare_nmf(m=200000, n=1000, r=30, rank=10)

    print("Running with larger rank...")
    compare_nmf(m=100000, n=5000, r=50, rank=20)
