# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-09 01:25:09 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark before any agent changes

**Result:** 10969.84 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #2 — 2026-06-09 01:26:47 UTC ✅ KEEP

**Hypothesis:** Replace baseline with optimized TriMul: fuse all 5 linear projections into a single matmul, use torch.bmm for the einsum contraction instead of einsum(), and eliminate the TriMul nn.Module overhead (no module instantiation per call). Removes unnecessary dtype conversions and model construction overhead.

**Result:** 8948.57 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #3 — 2026-06-09 01:27:30 UTC ❌ DISCARD

**Hypothesis:** Cast left/right to bfloat16 before bmm to use tensor cores, then cast back to float32. Expected speedup from tensor core throughput, but cast overhead outweighed the gain at these matrix sizes.

**Result:** 9007.42 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #4 — 2026-06-09 01:28:18 UTC ❌ DISCARD

**Hypothesis:** Use torch.matmul on [bs, H, N, N] tensors with explicit contiguous() calls instead of reshape to [bs*H, N, N] + bmm. Expected to avoid reshape copy overhead, but contiguous() on permuted tensors costs similarly. Marginally slower than best.

**Result:** 8964.66 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #5 — 2026-06-09 01:29:21 UTC ❌ DISCARD

**Hypothesis:** Use torch.compile(fullgraph=True, mode='reduce-overhead') to fuse elementwise ops and optimize memory access patterns. Expected kernel fusion to reduce memory bandwidth, but compile overhead and dynamic shapes across benchmark cases made it slower.

**Result:** 9011.89 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #6 — 2026-06-09 01:31:12 UTC ❌ DISCARD

**Hypothesis:** Use Triton fuse_gate_permute_kernel to fuse sigmoid+mask+permute into [bs*H,N,N] layout in one pass, avoiding separate permute+contiguous calls. Grid of bs*N*N blocks each writing H values with stride N*N. Much slower due to scattered writes and poor GPU occupancy vs simple PyTorch ops.

**Result:** 11226.95 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #7 — 2026-06-09 01:32:58 UTC ❌ DISCARD

**Hypothesis:** Interleave projection and gate weights as [left_proj, left_gate, right_proj, right_gate, out_gate] so adjacent weights are used together, hoping for better cache locality in downstream slicing. No improvement — the single fused matmul already processes all weights regardless of order.

**Result:** 9046.31 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #8 — 2026-06-09 01:33:51 UTC ❌ DISCARD

**Hypothesis:** Replace permute+reshape+bmm with torch.einsum('bikh,bjkh->bijh') hoping PyTorch dispatches to cuBLAS batched GEMM without explicit permute copies. Still slower — einsum overhead for 4D non-contiguous tensors is worse than explicit permute+bmm.

**Result:** 9030.87 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #9 — 2026-06-09 01:34:48 UTC ❌ DISCARD

**Hypothesis:** Split fused 5H matmul into 2 separate matmuls: [2H,dim] for left+left_gate and [3H,dim] for right+right_gate+out_gate. Avoids single large torch.cat but adds two smaller ones plus two matmul kernel launches. Net result: slightly slower than single fused matmul.

**Result:** 9004.98 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #10 — 2026-06-09 01:36:36 UTC ❌ DISCARD

**Hypothesis:** Custom CUDA kernels to fuse sigmoid+mask+permute into [bs*H,N,N] layout and permute [bs*H,N,N] back to [bs,N,N,H], avoiding PyTorch's copy-on-permute. Split into 2 matmuls [2H,dim] and [3H,dim]. Custom permute kernels help but extra matmul launches + torch.cat overhead keeps it slightly slower than best.

**Result:** 8992.78 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #11 — 2026-06-09 01:37:30 UTC ❌ DISCARD

**Hypothesis:** Combine single fused 5H matmul with custom CUDA permute kernels. Use interleaved [lp,lg,rp,rg,og] weight order so proj and gate are adjacent for fuse_to_bHNN. Still slower — .contiguous() on slices forces extra copies that negate the benefit.

**Result:** 9014.68 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #12 — 2026-06-09 01:39:06 UTC ❌ DISCARD

**Hypothesis:** Redesign CUDA kernel to read directly from full [bs,N,N,5H] proj tensor with configurable proj/gate offsets (stride total_H), avoiding .contiguous() slice copies. Non-contiguous strided reads hurt cache performance — worse than previous approach.

**Result:** 9037.35 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #13 — 2026-06-09 01:40:52 UTC ❌ DISCARD

**Hypothesis:** Use fp16 for projection matmul and bmm contraction — expected 2x tensor core throughput. Much slower: fp16 cast overhead for x, weights, mask, and the back-cast to fp32 adds substantial memory bandwidth that outweighs tensor core gains at these sizes.

**Result:** 11694.52 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #14 — 2026-06-09 01:42:15 UTC ❌ DISCARD

**Hypothesis:** Cache concatenated weight tensor across calls to avoid repeated torch.cat allocation. Module-level dict keyed on weight data_ptr. Catastrophically slow — likely because benchmark runs different configs that each trigger cache.clear() + torch.cat, plus the caching overhead itself. Also possible Modal container caching issue with previous fp16 code.

**Result:** 11526.00 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #15 — 2026-06-09 01:43:50 UTC ❌ DISCARD

**Hypothesis:** Work on flat [bs*N*N, H] proj slices and mask_flat=[bs*N*N,1] to avoid proj.reshape to 4D, then reshape+permute for bmm. Much slower — flat slice approach hurts memory access patterns and adds extra reshape steps. The 4D approach in #2 is better.

**Result:** 11380.21 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #16 — 2026-06-09 01:44:55 UTC ❌ DISCARD

**Hypothesis:** Restore exact #2 logic but use tensor method .sigmoid() instead of torch.sigmoid() to avoid dispatch overhead. Still showing ~11200µs — persistent Modal container issue inflating all recent benchmarks vs #2's 8948µs. The micro-optimization of .sigmoid() method vs torch.sigmoid() has no measurable effect.

**Result:** 11200.53 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #17 — 2026-06-09 01:46:19 UTC ❌ DISCARD

**Hypothesis:** Replace final F.linear with explicit torch.mm(out_flat, W.t()).reshape() to avoid F.linear overhead. Container issue resolved. Result 9004µs — slightly slower than #2's 8948µs. F.linear is already well-optimized and the explicit mm+reshape adds marginal overhead.

**Result:** 9004.29 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #18 — 2026-06-09 01:47:36 UTC ❌ DISCARD

**Hypothesis:** Cache concatenated weight tensor using data_ptr+H+dim tuple key to avoid torch.cat on every call. Python dict lookup overhead exceeds torch.cat savings — marginally slower than #2. The cat of 5 small weight tensors is cheaper than the dict overhead at benchmark speed.

**Result:** 9065.13 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #19 — 2026-06-09 01:49:08 UTC ❌ DISCARD

**Hypothesis:** Keep #2 logic but skip .contiguous() after final permute(0,2,3,1) to avoid a memory copy before layer_norm. Non-contiguous input to F.layer_norm is slightly slower than the contiguous path in #2 — layer_norm benefits from contiguous input even though it doesn't require it.

**Result:** 9063.72 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #20 — 2026-06-09 01:51:12 UTC ❌ DISCARD

**Hypothesis:** Triton kernel for N×N contraction reading [bs,N,N,H] directly without permute. Serial loop over k dimension in Triton kills performance for large N (N=1024 → 1024 serial iterations per thread block). cuBLAS batched GEMM (used by bmm after permute) is far more optimized than this naive tiled approach.

**Result:** 9154.03 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #21 — 2026-06-09 01:52:38 UTC ❌ DISCARD

**Hypothesis:** Replace F.linear(x_flat, all_weights) with torch.mm(x_flat, all_weights.t()) to avoid F.linear overhead. Worse: .t() creates a non-contiguous transpose view, causing cuBLAS to use a less optimal GEMM algorithm than F.linear's path which handles the transpose internally.

**Result:** 9377.15 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #22 — 2026-06-09 01:53:48 UTC ❌ DISCARD

**Hypothesis:** Pre-transpose weights to [dim, 5H] contiguous by cat-ing .t() views, then use torch.mm(x_flat, W_T) to avoid internal cuBLAS transpose. Still slightly slower than #2's F.linear: the torch.cat of transposed views does an extra copy to make [dim,5H] contiguous, negating any cuBLAS benefit.

**Result:** 9023.23 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #23 — 2026-06-09 01:55:27 UTC ❌ DISCARD

**Hypothesis:** Enable TF32 for matmuls (torch.backends.cuda.matmul.allow_tf32=True) to use H100 tensor cores at ~8x FP32 throughput. No improvement — TF32 is likely already the default on H100, or the benchmark harness controls this setting independently. Tolerance check passes (2% allowed).

**Result:** 9020.05 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #24 — 2026-06-09 01:56:51 UTC ❌ DISCARD

**Hypothesis:** Replace 3 separate sigmoid calls with one sigmoid over all 3H gate values (proj[...,2H:]) to reduce kernel launches. Much slower — single sigmoid over 3H reads/writes more data than 3 separate H-sized sigmoids, and the container may have a caching issue again.

**Result:** 11270.33 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #25 — 2026-06-09 01:58:04 UTC ❌ DISCARD

**Hypothesis:** Restore exact #2 code (fused 5H matmul + permute+bmm). Container was polluted by previous experiment (#24). This restores the clean baseline to verify the container is functional and confirms #2's approach is the best found. 9007µs vs #2's 8948µs — within measurement noise.

**Result:** 9006.93 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #26 — 2026-06-09 01:59:06 UTC ❌ DISCARD

**Hypothesis:** Cast left_t/right_t to bf16 AFTER permute+reshape (on contiguous [bs*H,N,N] tensors) for tensor core bmm, then cast result back to fp32. Different from #3 which cast before permute. Still slower than #2 — the bf16 cast+back-cast memory overhead exceeds tensor core speedup for these N×N matrix sizes.

**Result:** 9018.07 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

