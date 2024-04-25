import torch
from tensor_utils import roll_tensor


def gen_i_range_mask(B, I, J, K, i_lens, j_lens):
    indices_i = (
        torch.arange(I, device=i_lens.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, J)
    )
    indices_j = (
        torch.arange(J, device=i_lens.device).unsqueeze(0).unsqueeze(0).repeat(B, I, 1)
    )
    indices = indices_i + indices_j

    limit_s = (i_lens - 1).unsqueeze(-1).unsqueeze(-1).expand(B, I, J)
    limit_e = j_lens.unsqueeze(-1).unsqueeze(-1).expand(B, I, J)

    mask_b = (indices >= limit_s).flip(1)
    mask_e = (indices < limit_e).flip(1)

    mask = (mask_b & mask_e).unsqueeze(-1)
    diff = i_lens - i_lens.max()
    mask = roll_tensor(mask, shifts=diff, dim=1)

    bool_tensor = i_lens.unsqueeze(1) > torch.arange(I, device=i_lens.device)
    bool_tensor = bool_tensor[:, :, None, None].repeat(1, 1, J, 1)
    mask = mask * bool_tensor
    mask = mask.repeat(1, 1, 1, K)

    return mask


def gen_tri(B, I, J, K, device):
    triu = torch.triu(torch.ones((K, J), device=device), diagonal=0)
    triu = triu.unsqueeze(-1).unsqueeze(0)  # (1, K, J, 1)
    triu = triu.repeat(B, 1, 1, I)  # (B, K, J, I)
    triu = triu.transpose(1, 3)  # (B, I, J, K)
    return triu.bool()


def gen_most_i_mask(B, I, J, K, i_lens, j_lens, device):
    mask = torch.ones((B, I, J, K), dtype=torch.bool, device=device)
    for b in range(B):
        mask[b, i_lens[b] - 1, : j_lens[b] - 1] = False
    return mask


def get_invalid_tri_mask(B, I, J, K, text_mask, mel_mask, force_assign_last):
    i_lens = text_mask.sum(1)
    j_lens = mel_mask.sum(1)
    energy_mask = gen_i_range_mask(B, I, J, K, i_lens, j_lens)
    tri_ijk_mask = gen_tri(B, I, J, K, device=text_mask.device)
    if force_assign_last:
        most_i_mask = gen_most_i_mask(
            B, I, J, K, i_lens, j_lens, device=text_mask.device
        )
        return (~energy_mask) | (~tri_ijk_mask) | (~most_i_mask)
    else:
        return (~energy_mask) | (~tri_ijk_mask)

def pad_log_cond_prob_gt_backward(B, J, log_eps):
    x = torch.tril(torch.ones(B, 1, J-1, J-1), diagonal=1)
    x.masked_fill_(x == 0, log_eps)
    x.masked_fill_(x == 1, 0)
    return x

if __name__ == "__main__":
    # 测试用例1
    B, I, J, K = 2, 5, 10, 10
    i_lens = torch.tensor([5, 2])
    j_lens = torch.tensor([10, 5])
    masked_tensor = gen_i_range_mask(B, I, J, K, i_lens, j_lens).int()
    print(masked_tensor.shape)
