# lecture2_pytorch_resource_accounting.py
# -*- coding: utf-8 -*-
"""
CS336 Lecture 2: PyTorch & Resource Accounting -> Practical Python Notes

Covered:
- Floating point formats (fp32/fp16/bfloat16/fp8) & mixed precision
- PyTorch Tensor memory model: pointer, stride, views, contiguous vs. non-contiguous
- Creating tensors on GPU directly
- FLOPs accounting: linear layers, elementwise ops; forward/backward rough formulas
- MFU (Model FLOPs Utilization) estimator

References (line-cited to the provided PDF):
- Tensor as pointer, stride indexing, views & contiguous notes:  :contentReference[oaicite:10]{index=10}
- Create on GPU & FLOPs definitions/linear 2*B*D*K; forward ≈ 2*(tokens)*(params):  :contentReference[oaicite:11]{index=11}  :contentReference[oaicite:12]{index=12}
- MFU formula & typical ranges; forward/backward 2× / 4×:  :contentReference[oaicite:13]{index=13}  :contentReference[oaicite:14]{index=14}  :contentReference[oaicite:15]{index=15}  :contentReference[oaicite:16]{index=16}
- Floating formats & mixed precision highlights:  :contentReference[oaicite:17]{index=17}  :contentReference[oaicite:18]{index=18}  :contentReference[oaicite:19]{index=19}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import torch


# ---------------------------
# 1) Floating-point quick facts
# ---------------------------

DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float32: 4,     # fp32: 4 bytes
    torch.float16: 2,     # fp16: 2 bytes (narrow dynamic range)
    torch.bfloat16: 2,    # bfloat16: 2 bytes, fp32-like range, often better for DL forward
}
# torch.float8_* exists on some builds/hardware; keep optional names if present.
if hasattr(torch, "float8_e4m3fn"):
    DTYPE_BYTES[getattr(torch, "float8_e4m3fn")] = 1  # fp8 variant (H100-era support)
if hasattr(torch, "float8_e5m2"):
    DTYPE_BYTES[getattr(torch, "float8_e5m2")] = 1

def dtype_memory_footprint(numel: int, dtype: torch.dtype) -> int:
    """Return bytes to store a tensor with `numel` elements of given dtype."""
    return numel * DTYPE_BYTES[dtype]


# ---------------------------
# 2) Device helpers (CPU/GPU)
# ---------------------------

def pick_device() -> torch.device:
    """Prefer CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def make_zeros_on_device(shape: Tuple[int, ...], device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a tensor directly on a device (avoid host->device copies)."""
    device = device or pick_device()
    return torch.zeros(*shape, device=device, dtype=dtype)


# ---------------------------
# 3) Tensor: pointer, stride, views, contiguous
# ---------------------------

def explain_stride_and_views() -> None:
    """
    Demonstrate stride, views, and contiguous vs non-contiguous behavior.
    This function is safe to run on CPU; no GPU is required.
    """
    x = torch.arange(16).reshape(4, 4)  # shape [4,4]
    print("[x] shape:", x.shape, "stride:", x.stride(), "is_contiguous:", x.is_contiguous())

    # Transpose creates a non-contiguous view that shares storage
    xt = x.t()
    print("[x.t()] shape:", xt.shape, "stride:", xt.stride(), "is_contiguous:", xt.is_contiguous())

    # Many ops (like .view) require contiguous memory layout; use .contiguous() to get a copy
    try:
        _ = xt.view(16)  # Often raises due to non-contiguity
    except RuntimeError as e:
        print("[expected] .view on non-contiguous tensor raised:", str(e).splitlines()[0])

    # Correct approach:
    xtc = xt.contiguous().view(16)
    print("[fixed] xt.contiguous().view(16) ok, shape:", xtc.shape)

    # Index mapping example for a 2D tensor with strides (i,j) -> N
    # For a C-contiguous [4,4], stride is typically (4,1), so N = i*stride[0] + j*stride[1]
    i, j = 2, 3
    N = i * x.stride(0) + j * x.stride(1)
    print(f"[index-map] (i,j)=({i},{j}) -> flat index N={N}, check x.flatten()[N]==", x.flatten()[N].item())


# ---------------------------
# 4) FLOPs accounting
# ---------------------------

def flops_linear_forward(B: int, D: int, K: int) -> int:
    """
    FLOPs for y = x @ W where x:[B,D], W:[D,K].
    Each of the (B*K) outputs does D muls + D adds -> ~2*B*D*K FLOPs.
    """
    return 2 * B * D * K


def flops_elementwise(m: int, n: int) -> int:
    """O(m*n) elementwise FLOPs (e.g., add, relu)."""
    return m * n


def forward_backward_token_param_rules(num_tokens: int, num_params: int) -> Tuple[int, int]:
    """
    Lecture rough rules of thumb:
      forward  ≈ 2 * tokens * params
      backward ≈ 4 * tokens * params
    Returns (forward_flops, backward_flops).
    """
    fwd = 2 * num_tokens * num_params
    bwd = 4 * num_tokens * num_params
    return fwd, bwd


# ---------------------------
# 5) MFU (Model FLOPs Utilization)
# ---------------------------

@dataclass
class MFUStats:
    promised_flops_per_s: float  # e.g. vendor peak TFLOP/s (convert to FLOP/s)
    actual_flops_per_s: float    # measured FLOP/s from your training loop

def compute_mfu(actual_flops_per_s: float, promised_flops_per_s: float) -> float:
    """
    MFU = actual FLOP/s / promised FLOP/s
    Returns value in [0,1] (often << 1 in practice).
    """
    if promised_flops_per_s <= 0:
        return 0.0
    return actual_flops_per_s / promised_flops_per_s


# ---------------------------
# 6) Mixed precision quick demo (safe no-op if AMP unavailable)
# ---------------------------

def mixed_precision_demo(B: int = 8, D: int = 1024, K: int = 1024) -> None:
    """
    Illustrative demo: forward in bfloat16, specific ops in float32 if desired.
    This is a minimal example; real training would use autocast + GradScaler.
    """
    device = pick_device()
    x = torch.randn(B, D, device=device, dtype=torch.bfloat16)
    W = torch.randn(D, K, device=device, dtype=torch.bfloat16)

    # Example: do attention-like or numerically sensitive ops in float32 if needed
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        y = (x @ W).to(torch.float32)  # promote for a sensitive op (illustrative)
        # back to bfloat16 for the rest
        y = y.to(torch.bfloat16)
    print("[amp] y:", y.shape, y.dtype, "device:", y.device)


# ---------------------------
# 7) Create tensors on GPU directly (avoid H2D copies)
# ---------------------------

def create_on_gpu_example() -> None:
    device = pick_device()
    x_cpu = torch.zeros(32, 32)  # CPU
    y_gpu = x_cpu.to(device)     # copy to device
    print(f"[moved] x->y device: {y_gpu.device}")

    z_gpu = torch.zeros(32, 32, device=device)  # directly on device
    print(f"[direct] z device: {z_gpu.device}")


# ---------------------------
# 8) Quick self-check / example run
# ---------------------------

def _demo():
    print("== Floating-point bytes ==")
    for dt, b in DTYPE_BYTES.items():
        print(f"{str(dt):>20s} : {b} bytes/elem")

    print("\n== Stride & Views ==")
    explain_stride_and_views()

    print("\n== FLOPs Accounting ==")
    B, D, K = 64, 4096, 4096
    print("Linear forward FLOPs (2BDK):", flops_linear_forward(B, D, K))
    fwd, bwd = forward_backward_token_param_rules(num_tokens=B, num_params=(D*K))
    print("Rule-of-thumb forward/backward:", fwd, bwd)

    print("\n== MFU Example ==")
    # Suppose we measured actual 200 TFLOP/s on hardware promised 400 TFLOP/s:
    actual = 200e12
    promised = 400e12
    print("MFU:", compute_mfu(actual, promised))

    print("\n== Mixed Precision Demo ==")
    try:
        mixed_precision_demo(B=8, D=2048, K=2048)
    except RuntimeError as e:
        print("[warn] AMP or dtype not supported on this device/build:", str(e).splitlines()[0])

    print("\n== Create on GPU Example ==")
    create_on_gpu_example()


if __name__ == "__main__":
    _demo()
