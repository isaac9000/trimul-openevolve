# EVOLVE-BLOCK-START
import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    bs, seqlen, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]

    x = F.layer_norm(input_tensor.float(), [dim],
                     weights['norm.weight'].float(),
                     weights['norm.bias'].float())

    fused_w = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0).float()

    proj = x.reshape(-1, dim) @ fused_w.t()
    proj = proj.reshape(bs, seqlen, seqlen, 5 * hidden_dim)

    H = hidden_dim
    left_p  = proj[..., :H]
    right_p = proj[..., H:2*H]
    left_g  = proj[..., 2*H:3*H]
    right_g = proj[..., 3*H:4*H]
    out_g   = proj[..., 4*H:5*H]

    mask_e = mask.unsqueeze(-1)
    left  = left_p  * torch.sigmoid(left_g)  * mask_e
    right = right_p * torch.sigmoid(right_g) * mask_e
    out_gate = torch.sigmoid(out_g)

    # Contraction via bmm in bf16: (bs*H, N, N) x (bs*H, N, N)^T
    left_t  = left.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * H, seqlen, seqlen)
    right_t = right.to(torch.bfloat16).permute(0, 3, 1, 2).reshape(bs * H, seqlen, seqlen)
    out_bmm = torch.bmm(left_t, right_t.transpose(1, 2))
    out_bmm = out_bmm.reshape(bs, H, seqlen, seqlen).permute(0, 2, 3, 1).float()

    out_n = F.layer_norm(out_bmm, [H],
                         weights['to_out_norm.weight'].float(),
                         weights['to_out_norm.bias'].float())
    out_n = out_n * out_gate

    result = out_n.reshape(-1, H) @ weights['to_out.weight'].float().t()
    return result.reshape(bs, seqlen, seqlen, dim)
# EVOLVE-BLOCK-END
