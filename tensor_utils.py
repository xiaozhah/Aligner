import torch


def roll_tensor(tensor, shifts, dim):
    # 获取tensor的形状
    shape = tensor.size()

    # 确保dim在有效范围内
    assert dim >= 0 and dim < len(shape), "Invalid dimension"

    # 生成一个索引tensor
    indices = (
        torch.arange(shape[dim], device=tensor.device)
        .view([1] * dim + [-1] + [1] * (len(shape) - dim - 1))
        .expand(shape)
    )

    # 将shifts转换为tensor并调整形状
    shifts = shifts.view([-1] + [1] * (len(shape) - 1))

    # 计算移位后的索引
    shifted_indices = (indices - shifts) % shape[dim]

    # 使用移位后的索引对tensor进行索引操作
    result = tensor.gather(dim, shifted_indices.expand(shape))

    return result


def roll_tensor_1d(tensor, shifts):
    # 获取tensor的形状
    shape = tensor.size()
    dim = 1  # 沿着第二个维度进行移位

    # 确保dim在有效范围内
    assert dim >= 0 and dim < len(shape), "Invalid dimension"

    # 生成一个索引tensor
    indices = (
        torch.arange(shape[dim])
        .view([1] * dim + [-1] + [1] * (len(shape) - dim - 1))
        .expand(shape)
    )

    # 将shifts转换为tensor并调整形状
    shifts = shifts.view([-1] + [1] * (len(shape) - 1))

    # 计算移位后的索引
    shifted_indices = (indices - shifts) % shape[dim]

    # 使用移位后的索引对tensor进行索引操作
    result = tensor.gather(dim, shifted_indices.expand(shape))

    return result


def right_shift(x, shifts_text_dim, shifts_mel_dim):
    """
    Shift the tensor x to the right along the text and mel dimensions.

    Args:
        x (torch.Tensor): The input tensor of shape (B, I, J, K).
        shifts_text_dim (torch.Tensor): The shift amounts along the text dimension of shape (B,).
        shifts_mel_dim (torch.Tensor): The shift amounts along the mel dimension of shape (B,).

    Returns:
        torch.Tensor: The right-shifted tensor of shape (B, I, J, K).
    """
    x = roll_tensor(x, shifts=shifts_text_dim, dim=1)
    x = roll_tensor(x, shifts=shifts_mel_dim, dim=2)
    return x


def left_shift(x, shifts_text_dim, shifts_mel_dim):
    """
    Shift the tensor x to the left along the text and mel dimensions.

    Args:
        x (torch.Tensor): The input tensor of shape (B, I, J).
        shifts_text_dim (torch.Tensor): The shift amounts along the text dimension of shape (B,).
        shifts_mel_dim (torch.Tensor): The shift amounts along the mel dimension of shape (B,).

    Returns:
        torch.Tensor: The left-shifted tensor of shape (B, I, J).
    """
    x = x.unsqueeze(-1)
    x = roll_tensor(x, shifts=-shifts_text_dim, dim=1)
    x = roll_tensor(x, shifts=-shifts_mel_dim, dim=2)
    x = x.squeeze(-1)
    return x


class LinearNorm(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, w_init_gain="linear", weight_norm=False
    ):
        super(LinearNorm, self).__init__()
        if weight_norm:
            self.linear_layer = torch.nn.utils.weight_norm(
                torch.nn.Linear(in_dim, out_dim, bias=bias)
            )
        else:
            self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


if __name__ == "__main__":
    # 示例用法 1
    tensor1 = torch.tensor(
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ]
    )
    shifts1 = torch.tensor([1, 2])  # 每个sample在最后一个维度上的右移量
    dim1 = 3  # 在最后一个维度上进行移位
    result1 = roll_tensor(tensor1, shifts1, dim1)
    print("示例 1 - 在最后一个维度上移位:")
    print(result1)

    # 示例用法 2
    tensor2 = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
        ]
    )
    shifts2 = torch.tensor([1, 2, 3])  # 每个sample在第二个维度上的左移量
    dim2 = 1  # 在第二个维度上进行移位
    result2 = roll_tensor(tensor2, shifts2, dim2)
    print("示例 2 - 在第二个维度上移位:")
    print(result2)

    # 示例用法
    tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    shifts = torch.tensor([1, 2])  # 第一行右移1个位置,第二行右移2个位置

    result = roll_tensor_1d(tensor, shifts)
    print(result)
