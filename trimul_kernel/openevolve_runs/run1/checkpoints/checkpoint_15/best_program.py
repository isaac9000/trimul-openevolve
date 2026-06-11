# EVOLVE-BLOCK-START
"""
Optimized TriMul: fused projections, bf16 bmm contraction, minimal overhead.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    bs, N, _, dim = input_tensor.shape
    H = config["hidden_dim"]

    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    ton_w = weights['to_out_norm.weight']
    ton_b = weights['to_out_norm.bias']
    to_w = weights['to_out.weight']  # [dim, H]

    # Fuse 5 projections into one matmul
    fused_w = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)  # [5H, dim]

    # LayerNorm then project in bf16 for faster tensor core GEMM
    x = F.layer_norm(input_tensor, [dim], norm_w, norm_b)  # [bs, N, N, dim]
    x_flat = x.reshape(-1, dim).to(torch.bfloat16)
    fused_w_bf16 = fused_w.to(torch.bfloat16)

    # Fused projection in bf16
    proj = F.linear(x_flat, fused_w_bf16).reshape(bs, N, N, 5 * H)  # bf16

    # Compute out_gate in fp32 for accuracy
    og = proj[..., 4*H:5*H].to(torch.float32).sigmoid()

    # Apply gates and mask - stay in bf16
    m = mask.to(torch.bfloat16).unsqueeze(-1)  # [bs, N, N, 1]
    left = proj[..., :H] * proj[..., 2*H:3*H].sigmoid() * m   # [bs, N, N, H] bf16
    right = proj[..., H:2*H] * proj[..., 3*H:4*H].sigmoid() * m  # [bs, N, N, H] bf16

    # Contraction: permute to [bs*H, N, N] for batched matmul
    left_b = left.permute(0, 3, 1, 2).reshape(bs * H, N, N)
    right_b = right.permute(0, 3, 1, 2).reshape(bs * H, N, N)
    out_b = torch.bmm(left_b, right_b.mT)  # [bs*H, N, N] bf16

    # Reshape back to [bs, N, N, H] and cast to fp32 for layernorm
    out = out_b.reshape(bs, H, N, N).permute(0, 2, 3, 1).to(torch.float32)

    # to_out_norm + out_gate + to_out
    out = F.layer_norm(out, [H], ton_w, ton_b) * og
    out = F.linear(out, to_w)  # [bs, N, N, dim]

    return out
# EVOLVE-BLOCK-END
