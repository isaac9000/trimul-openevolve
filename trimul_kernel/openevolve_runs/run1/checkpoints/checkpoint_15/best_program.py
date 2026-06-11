# EVOLVE-BLOCK-START
"""
Optimized TriMul: fused 5-way projection, efficient bmm contraction in bf16.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device

    bs, N, _, D = input_tensor.shape
    H = hidden_dim

    norm_w = weights['norm.weight'].to(torch.float32)
    norm_b = weights['norm.bias'].to(torch.float32)
    on_w = weights['to_out_norm.weight'].to(torch.float32)
    on_b = weights['to_out_norm.bias'].to(torch.float32)
    to_w = weights['to_out.weight'].to(torch.bfloat16)  # [D, H]

    # Fused weight in bf16 for tensor cores
    fused_w = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).to(torch.bfloat16)  # [5H, D]

    # LayerNorm in fp32
    x = F.layer_norm(input_tensor.to(torch.float32), [D], norm_w, norm_b)

    # Fused projection in bf16
    x_bf16 = x.to(torch.bfloat16).reshape(-1, D)
    proj = (x_bf16 @ fused_w.t()).reshape(bs, N, N, 5 * H)  # bf16

    # Gates and projections - keep bf16 throughout
    lp = proj[..., :H]
    rp = proj[..., H:2*H]
    lg = torch.sigmoid(proj[..., 2*H:3*H].float()).to(torch.bfloat16)
    rg = torch.sigmoid(proj[..., 3*H:4*H].float()).to(torch.bfloat16)
    out_gate = torch.sigmoid(proj[..., 4*H:5*H].float())  # keep fp32 for final mul

    mask_e = mask.unsqueeze(-1).to(torch.bfloat16)
    left  = (lp * lg * mask_e).permute(0, 3, 1, 2).contiguous().reshape(bs * H, N, N)
    right = (rp * rg * mask_e).permute(0, 3, 1, 2).contiguous().reshape(bs * H, N, N)

    # Contraction: [bs*H, N, N] x [bs*H, N, N]^T -> [bs*H, N, N]
    out_bh = torch.bmm(left, right.transpose(1, 2))

    # Reshape: [bs, H, N, N] -> [bs, N, N, H] in fp32
    out = out_bh.reshape(bs, H, N, N).permute(0, 2, 3, 1).to(torch.float32)

    # LayerNorm + out_gate + final projection
    out = F.layer_norm(out, [H], on_w, on_b)
    out = (out * out_gate).to(torch.bfloat16).reshape(-1, H)
    out = (out @ to_w.t()).reshape(bs, N, N, D)

    return out.to(torch.float32)
# EVOLVE-BLOCK-END
