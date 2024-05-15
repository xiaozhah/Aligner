import torch
import torch.nn.functional as F
import numpy as np


def roll_tensor(tensor, shifts, dim):
    """
    Right shift the multi-dimensional tensor (2D or 3D or 4D...) along the given dimension.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, I) or (B, I, J) or (B, I, J, K) or ....
        shifts (torch.Tensor): The shift amounts along the dimension of shape (B,)
        dim (int): The dimension to roll.

    Returns:
        result (torch.Tensor): The rolled tensor of shape (B, I) or (B, I, J) or (B, I, J, K) or ....
    """

    # 获取tensor的形状
    shape = tensor.size()

    # 确保dim在有效范围内
    assert dim >= 0 and dim < len(shape), "Invalid hidden"

    # 生成一个索引tensor
    indices = (
        torch.arange(shape[dim], device=tensor.device)
        .view([1] * dim + [-1] + [1] * (len(shape) - dim - 1))
        .expand(shape)
    )

    # 调整shifts形状
    shifts = shifts.view([-1] + [1] * (len(shape) - 1))

    # 计算移位后的索引
    shifted_indices = (indices - shifts) % shape[dim]

    # 使用移位后的索引对tensor进行索引操作
    result = tensor.gather(dim, shifted_indices.expand(shape))

    return result


def shift_tensor(x, shifts_text_dim, shifts_mel_dim):
    """
    Shift the tensor x to the right along the text and mel hiddens.

    Args:
        x (torch.Tensor): The input tensor of shape (B, I, J, K).
        shifts_text_dim (torch.Tensor): The shift amounts along the text hidden of shape (B,).
        shifts_mel_dim (torch.Tensor): The shift amounts along the mel hidden of shape (B,).

    Returns:
        x (torch.Tensor): The right-shifted tensor of shape (B, I, J, K).
    """
    x = roll_tensor(x, shifts=shifts_text_dim, dim=1)
    x = roll_tensor(x, shifts=shifts_mel_dim, dim=2)
    return x


def one_hot(B, I, device):
    """
    Generate a one-hot vector of shape (I,) on the device.
    """
    x = torch.full((I,), -float("inf"), device=device)
    x[0] = 0
    return x


def reverse_and_pad_head_tail_on_alignment(
    log_boundary_backward, text_mask_backward, mel_mask_backward
):
    """
    Reverse the alignment and pad the boundary matrix.

    Args:
        log_boundary_backward (torch.Tensor): The log boundary matrix of shape (B, I, J-2).
        text_mask_backward (torch.Tensor): The text mask of shape (B, I-1).
        mel_mask_backward (torch.Tensor): The mel spectrogram mask of shape (B, J-1).

    Returns:
        log_boundary_backward (torch.Tensor): The reversed and padded alignment matrix of shape (B, I, J).
    """
    B, I, _ = log_boundary_backward.shape

    onehot = one_hot(B, I, device=log_boundary_backward.device)[None, :, None].repeat(
        B, 1, 1
    )
    # (B, I, J-2) -> (B, I, J-1)
    log_boundary_backward = torch.cat((onehot, log_boundary_backward), dim=2)

    # mel index: (J-1)-(j-1)+1 = J-j+1 -> 1, means shift iter num = J-j
    # text index: I-i+1 -> 1, means shift iter num = I-i
    shifts_text_dim = compute_max_length_diff(text_mask_backward)
    shifts_mel_dim = compute_max_length_diff(mel_mask_backward)
    log_boundary_backward = shift_tensor(
        log_boundary_backward.flip(1, 2),
        shifts_text_dim=-shifts_text_dim,
        shifts_mel_dim=-shifts_mel_dim,
    )
    log_boundary_backward = torch.cat((onehot, log_boundary_backward), dim=2)
    return log_boundary_backward


def compute_max_length_diff(mask):
    # returned value is always >= 0
    lengths = mask.sum(1)
    return lengths.max() - lengths


def arange_from_0(lens: torch.LongTensor):
    B = len(lens)
    I = lens.max()
    x = torch.arange(I).unsqueeze(0).repeat(B, 1)
    mask = x >= lens.unsqueeze(1)
    x.masked_fill_(mask, 0)
    return x


def arange_to_end(i_lens: torch.LongTensor, j_lens: torch.LongTensor):
    B = len(i_lens)
    I = i_lens.max()
    strat = j_lens - i_lens
    x = torch.arange(I).unsqueeze(0).repeat(B, 1) + strat.unsqueeze(1)
    mask = x >= j_lens.unsqueeze(1)
    x.masked_fill_(mask, 0)
    return x


def gen_i_range_mask(B, I, D, J, text_mask, mel_mask):
    """
    Generate a mask which limit the boundary index range of mel.

    Args:
        B (int): The batch size.
        I (int): The number of text hiddens.
        D (int): The number of candidate mel hiddens.
        J (int): The number of mel hiddens.
        text_mask (torch.Tensor): The text mask of shape (B, I).
        mel_mask (torch.Tensor): The mel mask of shape (B, J).

    Returns:
        mask (torch.Tensor): The mask of shape (B, I, D, J).
    """
    i_lens = text_mask.sum(1)
    j_lens = mel_mask.sum(1)

    indices_d = torch.arange(D, device=i_lens.device)[:, None]
    indices_j = torch.arange(J, device=i_lens.device)[None, :]
    indices = (indices_d + indices_j).unsqueeze(0).unsqueeze(0).repeat(B, I, 1, 1)

    mask_b = indices > (arange_from_0(i_lens) - 1)[:, :, None, None]
    mask_e = indices <= arange_to_end(i_lens, j_lens)[:, :, None, None]

    mask = mask_b & mask_e

    mask = mask * text_mask.unsqueeze(-1).unsqueeze(-1)

    return mask


def convert_geq_to_gt(log_cond_prob_geq_backward):
    """
    "greater than or equal to" format to "greater than" format

    Args:
        log_cond_prob_geq_backward (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I-1, J-1, J-1).

    Returns:
        log_cond_prob_geq_backward (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I-1, J-2, J-1).
    """
    return log_cond_prob_geq_backward[:, :, 1:]


def gt_pad_on_text_dim(log_cond_prob_gt_backward, text_mask, log_eps):
    """
    pad the last text hidden which using prior knowledge for "greater than" format

    Args:
        log_cond_prob_gt_backward (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I-1, J-2, J-1).
        text_mask (torch.Tensor): The text mask of shape (B, I).

    Returns:
        log_cond_prob_gt_backward (torch.Tensor): The padded log cumulative conditional probability tensor of shape (B, I, J-2, J-1).
    """

    # (B, I-1, J-2, J-1) -> (B, I, J-2, J-1)
    log_cond_prob_gt_backward = F.pad(
        log_cond_prob_gt_backward, (0, 0, 0, 0, 0, 1), "constant", log_eps
    )

    # (B, I, J-2, J-1) -> (B, I, J-2, J-1)
    log_cond_prob_gt_backward = geq_mask_on_text_dim(
        log_cond_prob_gt_backward, text_mask
    )

    return log_cond_prob_gt_backward


def geq_mask_on_text_dim(log_cond_prob_geq_or_gt, text_mask):
    """
    pad the last text hidden which using prior knowledge for "greater than or equal to" format

    Args:
        log_cond_prob_geq_or_gt (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I, J, J) for forward, or (B, I, J-2, J-1) for backward.
        text_mask (torch.Tensor): The text mask of shape (B, I).

    Returns:
        log_cond_prob_geq_or_gt (torch.Tensor): The padded log cumulative conditional probability tensor of shape (B, I, J, J) for forward, or (B, I, J-2, J-1) for backward.
    """
    B, _, J, K = log_cond_prob_geq_or_gt.shape
    diagonal = 0 if J == K else 1  # if forward else backward
    pad = torch.tril(
        torch.ones(J, K, device=log_cond_prob_geq_or_gt.device), diagonal=diagonal
    )
    mask = torch.zeros_like(log_cond_prob_geq_or_gt)
    i_lens = text_mask.sum(1)
    mask[torch.arange(B), i_lens - 1, :, :] = pad
    log_cond_prob_geq_or_gt.masked_fill_(mask.bool(), 0)
    return log_cond_prob_geq_or_gt


def get_valid_max(tensor, mask, inf_value=1e6):
    """
    Calculate the minimum and maximum values of the valid elements in the given 2D tensor.

    Args:
        tensor: 2D tensor, shape (B, L)
        mask: 2D mask tensor, shape (B, L), valid elements are 1, invalid elements are 0
        inf_value: The value to use for masking invalid elements.

    Returns:
        min_values: The minimum value of the valid elements in each sample, shape (B,)
        max_values: The maximum value of the valid elements in each sample, shape (B,)
    """
    masked_tensor = tensor.masked_fill(~mask, inf_value)
    min_values, _ = torch.min(masked_tensor, dim=1)

    masked_tensor = tensor.masked_fill(~mask, -inf_value)
    max_values, _ = torch.max(masked_tensor, dim=1)

    return min_values, max_values


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_padding_mask(lens, max_lens=None):
    bsz = lens.size(0)
    if max_lens is None:
        max_lens = torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_mask(lens, max_lens=None):
    return ~lengths_to_padding_mask(lens, max_lens)


def get_mat_p_f(src_tokens, durations):
    """
    Calculate the mapping matrix (mat_p_f) from the text tokens, e.g. phone, to the mel spectrograms.

    Args:
        src_tokens (torch.Tensor): The input tensor of shape (B, L, C).
        durations (torch.Tensor): The duration tensor of shape (B, L).

    Returns:
        mat_p_f (torch.Tensor): The mapping matrix of shape (B, L, T).
    """
    assert (
        src_tokens.shape[:2] == durations.shape
    ), "src_tokens and durations should have the same batch size and length"
    B, L, _ = src_tokens.shape
    T = durations.sum(axis=-1).max()
    cumsum_dur_1 = torch.cumsum(durations, dim=-1)  # [B, L]
    cumsum_dur_0 = cumsum_dur_1 - durations  # [B, L]

    mask1 = lengths_to_mask(cumsum_dur_1.flatten(), T).reshape(B, L, T)
    mask0 = lengths_to_mask(cumsum_dur_0.flatten(), T).reshape(B, L, T)
    mat_p_f = (mask1 & ~mask0).float()
    return mat_p_f


def calculate_tensor_memory_size(shape, dtype):
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    dtype_size = torch.tensor([], dtype=dtype, device="cpu").element_size()
    memory_size_in_bytes = total_elements * dtype_size
    memory_size_in_mb = memory_size_in_bytes / (1024 * 1024)
    return memory_size_in_mb


def cal_max_hidden_memory_size(selected_boundary_indices, text_mask):
    B, K = selected_boundary_indices.shape
    _, I = text_mask.shape
    shape = (B, I, K, K)  # max tensor shape in MoBo aligner
    dtype = torch.float32
    memory_size_mb = calculate_tensor_memory_size(shape, dtype)
    print(
        f"Memory size for tensor with shape {shape} and dtype {dtype}: {memory_size_mb:.2f} MB"
    )


def diag_logsumexp(tensor, from_ind):
    """
    Calculate the logsumexp of the diagonals of a 3D tensor, from the diagonal with index from_ind to the main diagonal.

    Args:
        tensor: A 3D tensor of shape (B, I, J).
        from_ind: The index of the diagonal to start from.

    Returns:
        A 2D tensor of shape (B, J) containing the logsumexp of the diagonals of the input tensor.
    """
    batch_size, dim1, dim2 = tensor.size()

    assert from_ind < dim2, "from_ind should be less than dim2"

    logsumexp_result = torch.full(
        (batch_size, dim2 - from_ind), -float("inf"), device=tensor.device
    )

    for i in range(from_ind, dim2):
        diagonal_elements = tensor.diagonal(offset=i - dim1 + 1, dim1=-2, dim2=-1)
        logsumexp_result[:, i - from_ind] = torch.logsumexp(diagonal_elements, dim=-1)

    return logsumexp_result


if __name__ == "__main__":
    # 示例用法 1 (4D tensor)
    tensor1 = torch.tensor(
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ]
    )  # shape is (2, 2, 2, 3)
    shifts1 = torch.tensor([1, 2])  # 每个sample在最后一个维度上的右移量
    dim1 = 3  # 在最后一个维度上进行移位
    result1 = roll_tensor(tensor1, shifts1, dim1)
    print("示例 1 - 在最后一个维度上移位:")
    print(result1)

    # 示例用法 2 (3D tensor)
    tensor2 = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
        ]
    )  # shape is (3, 3, 2)
    shifts2 = torch.tensor([1, 2, 3])  # 每个sample在第二个维度上的左移量
    dim2 = 1  # 在第二个维度上进行移位
    result2 = roll_tensor(tensor2, shifts2, dim2)
    print("示例 2 - 在第二个维度上移位:")
    print(result2)

    # 示例用法 3 (2D tensor)
    tensor3 = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )  # shape is (3, 3)
    shifts3 = torch.tensor([1, 2, 3])
    dim3 = 1
    result3 = roll_tensor(tensor3, shifts3, dim3)
    print("示例 3 - 在第一个维度上移位:")
    print(result3)

    # 测试用例 4
    B, I, D, J = 2, 5, 10, 16
    text_mask = torch.ones(2, 5)
    text_mask[1, 2:] = 0
    mel_mask = torch.ones(2, 16)
    mel_mask[1, 5:] = 0
    print("示例 4 - 生成_i_range_mask")
    masked_tensor = gen_i_range_mask(B, I, D, J, text_mask, mel_mask).int()
    print(masked_tensor)

    # 测试用例 5
    i_lens = text_mask.sum(1)
    j_lens = mel_mask.sum(1)
    print("示例 5 - arange_from_0")
    print(arange_from_0(i_lens))
    print("示例 5 - arange_to_end")
    print(arange_to_end(i_lens, j_lens))

    # 测试用例 6
    tensor = torch.tensor(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ]
    )

    # 调用函数计算副对角线的logsumexp并获取结果tensor
    result = diag_logsumexp(tensor.flip(1), from_ind=3)
    print(result)
