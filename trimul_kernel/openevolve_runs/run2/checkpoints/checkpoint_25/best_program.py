# EVOLVE-BLOCK-START
"""
Optimized TriMul: fused projections + bf16 bmm contraction.
"""

import torch
import torch.nn.functional as F

# Enable TF32 tensor cores for matmul/bmm — large speedup on H100 with
# accuracy far better than bf16 (10-bit vs 8-bit mantissa), typically
# within fp32 test tolerances.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


_cache = {}


def _get_fused(weights, dim, hidden_dim, device):
    # Use a robust cache key based on tensor identity of a representative weight
    key = (weights['left_proj.weight'].data_ptr(),
           weights['to_out.weight'].data_ptr(), dim, hidden_dim)
    if key in _cache:
        return _cache[key]
    # Stack the 5 projection weights into one [5*hidden_dim, dim] matrix
    # order: left_proj, right_proj, left_gate, right_gate, out_gate
    W = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).to(device=device, dtype=torch.float32).contiguous()

    norm_w = weights['norm.weight'].to(device=device, dtype=torch.float32)
    norm_b = weights['norm.bias'].to(device=device, dtype=torch.float32)
    on_w = weights['to_out_norm.weight'].to(device=device, dtype=torch.float32)
    on_b = weights['to_out_norm.bias'].to(device=device, dtype=torch.float32)
    out_w = weights['to_out.weight'].to(device=device, dtype=torch.float32).contiguous()

    res = (W, norm_w, norm_b, on_w, on_b, out_w)
    _cache[key] = res
    return res


def _contract(left, right, bs, H, N):
    # out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d]
    # permute to [b,d,i,k] @ [b,d,j,k]^T = [b,d,i,j]
    # TF32 bmm (fp32 storage, tensor-core math) — fast and accurate (10-bit
    # mantissa vs 8-bit for bf16), keeping us within fp32 test tolerances.
    left_b = left.permute(0, 3, 1, 2).reshape(bs * H, N, N).contiguous()
    right_b = right.permute(0, 3, 1, 2).reshape(bs * H, N, N).contiguous()
    out_b = torch.bmm(left_b, right_b.transpose(1, 2))
    return out_b.reshape(bs, H, N, N).permute(0, 2, 3, 1)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device

    bs, N, _, _ = input_tensor.shape

    W, norm_w, norm_b, on_w, on_b, out_w = _get_fused(weights, dim, hidden_dim, device)

    x = input_tensor.to(torch.float32)
    # LayerNorm over last dim
    x = F.layer_norm(x, (dim,), norm_w, norm_b)

    # Fused projection: [bs, N, N, dim] @ W.T -> [bs, N, N, 5*hidden_dim]
    proj = F.linear(x, W)  # [bs, N, N, 5H]
    left, right, lg, rg, og = proj.split(hidden_dim, dim=-1)

    m = mask.unsqueeze(-1)
    # Fold mask into the gated projections. Combine sigmoid(gate)*mask first
    # so the elementwise scaling is a single fused pass over the projection.
    left = left * torch.sigmoid(lg).mul_(m)
    right = right * torch.sigmoid(rg).mul_(m)
    out_gate = torch.sigmoid(og)

    # Contraction: einsum('b i k d, b j k d -> b i j d')
    H = hidden_dim
    out = _contract(left, right, bs, H, N)

    out = F.layer_norm(out.contiguous(), (hidden_dim,), on_w, on_b)
    out.mul_(out_gate)
    output = F.linear(out, out_w)

    return output
# EVOLVE-BLOCK-END
