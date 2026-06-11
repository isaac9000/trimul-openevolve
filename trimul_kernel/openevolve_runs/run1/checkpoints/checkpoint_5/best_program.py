# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    bs, N, _, _ = input_tensor.shape
    device = input_tensor.device

    # LayerNorm input
    norm_w = weights['norm.weight'].float()
    norm_b = weights['norm.bias'].float()
    x = F.layer_norm(input_tensor.float(), [dim], norm_w, norm_b)

    # Fuse 5 projections into one matmul: [lp, rp, lg, rg, og] stacked
    # weights shape: (hidden_dim, dim) each
    lp_w = weights['left_proj.weight'].float()
    rp_w = weights['right_proj.weight'].float()
    lg_w = weights['left_gate.weight'].float()
    rg_w = weights['right_gate.weight'].float()
    og_w = weights['out_gate.weight'].float()

    # Fused weight: (5*hidden_dim, dim) in bfloat16 for tensor cores
    fused_w = torch.cat([lp_w, rp_w, lg_w, rg_w, og_w], dim=0).to(torch.bfloat16)

    # x: [bs, N, N, dim] -> reshape to [bs*N*N, dim]
    x_flat = x.to(torch.bfloat16).reshape(-1, dim)
    # fused proj: [bs*N*N, 5*hidden_dim] using tensor cores
    proj = F.linear(x_flat, fused_w).float()
    proj = proj.reshape(bs, N, N, 5 * hidden_dim)

    left_proj  = proj[..., :hidden_dim]
    right_proj = proj[..., hidden_dim:2*hidden_dim]
    left_gate  = torch.sigmoid(proj[..., 2*hidden_dim:3*hidden_dim])
    right_gate = torch.sigmoid(proj[..., 3*hidden_dim:4*hidden_dim])
    out_gate   = torch.sigmoid(proj[..., 4*hidden_dim:5*hidden_dim])

    # Apply mask and gates
    m = mask.unsqueeze(-1)  # [bs, N, N, 1]
    left  = left_proj  * left_gate  * m   # [bs, N, N, H]
    right = right_proj * right_gate * m   # [bs, N, N, H]

    # Contraction: out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d]
    # Reshape: left -> [bs*H, N, N], right -> [bs*H, N, N]
    # out = left @ right^T -> [bs*H, N, N]
    # Use bfloat16 for tensor core acceleration
    left_bf  = left.to(torch.bfloat16).permute(0, 3, 1, 2).contiguous().reshape(bs * hidden_dim, N, N)
    right_bf = right.to(torch.bfloat16).permute(0, 3, 1, 2).contiguous().reshape(bs * hidden_dim, N, N)
    # bmm: [bs*H, N, N] x [bs*H, N, N]^T -> [bs*H, N, N]
    out_bmm = torch.bmm(left_bf, right_bf.transpose(1, 2)).float()
    # out_bmm: [bs*H, N, N] -> [bs, H, N, N] -> [bs, N, N, H]
    out = out_bmm.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1)

    # to_out_norm + out_gate + to_out
    ton_w = weights['to_out_norm.weight'].float()
    ton_b = weights['to_out_norm.bias'].float()
    out = F.layer_norm(out, [hidden_dim], ton_w, ton_b)
    out = out * out_gate

    to_out_w = weights['to_out.weight'].to(torch.bfloat16)
    out = F.linear(out.to(torch.bfloat16), to_out_w).float()

    return out
# EVOLVE-BLOCK-END
