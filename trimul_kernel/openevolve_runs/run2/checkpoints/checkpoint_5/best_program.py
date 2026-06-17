# EVOLVE-BLOCK-START
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


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device

    bs, N, _, _ = input_tensor.shape

    # Weights (float32)
    nw = weights['norm.weight'].to(torch.float32)
    nb = weights['norm.bias'].to(torch.float32)
    lp = weights['left_proj.weight'].to(torch.float32)
    rp = weights['right_proj.weight'].to(torch.float32)
    lg = weights['left_gate.weight'].to(torch.float32)
    rg = weights['right_gate.weight'].to(torch.float32)
    og = weights['out_gate.weight'].to(torch.float32)
    onw = weights['to_out_norm.weight'].to(torch.float32)
    onb = weights['to_out_norm.bias'].to(torch.float32)
    ow = weights['to_out.weight'].to(torch.float32)

    # Fuse the 5 projections into one weight matrix [5*hidden_dim, dim]
    fused_w = torch.cat([lp, rp, lg, rg, og], dim=0)  # [5H, dim]

    # LayerNorm on input
    x = torch.nn.functional.layer_norm(input_tensor, (dim,), nw, nb)  # [bs,N,N,dim]

    # Fused projection: [bs,N,N,dim] x [dim,5H] -> [bs,N,N,5H]
    proj = torch.nn.functional.linear(x, fused_w)  # [bs,N,N,5H]

    H = hidden_dim
    left, right, left_g, right_g, out_g = proj.split(H, dim=-1)

    m = mask.unsqueeze(-1)
    left = left * m * torch.sigmoid(left_g)
    right = right * m * torch.sigmoid(right_g)

    # Contraction: einsum('b i k d, b j k d -> b i j d')
    # Permute to [bs, d, i, k] and [bs, d, k, j] then bmm
    lb = left.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * H, N, N)   # [bs*H, i, k]
    rb = right.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * H, N, N)  # [bs*H, j, k]
    out = torch.bmm(lb, rb.transpose(1, 2))  # [bs*H, i, j]
    out = out.reshape(bs, H, N, N).permute(0, 2, 3, 1).to(torch.float32)  # [bs,N,N,H]

    out = torch.nn.functional.layer_norm(out, (H,), onw, onb)
    out = out * torch.sigmoid(out_g)
    output = torch.nn.functional.linear(out, ow)

    return output
# EVOLVE-BLOCK-END
