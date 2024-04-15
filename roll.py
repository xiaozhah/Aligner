import torch

def roll_tensor(tensor, shifts, dim):
    # 获取tensor的形状
    shape = tensor.size()
    
    # 确保dim在有效范围内
    assert dim >= 0 and dim < len(shape), "Invalid dimension"
    
    # 生成一个索引tensor
    indices = torch.arange(shape[dim]).view([1] * dim + [-1] + [1] * (len(shape) - dim - 1)).expand(shape)
    
    # 将shifts转换为tensor并调整形状
    shifts = torch.tensor(shifts).view([-1] + [1] * (len(shape) - 1))
    
    # 计算移位后的索引
    shifted_indices = (indices - shifts) % shape[dim]
    
    # 使用移位后的索引对tensor进行索引操作
    result = tensor.gather(dim, shifted_indices.expand(shape))
    
    return result

if __name__ == "__main__":
    # 示例用法 1
    tensor1 = torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]])
    shifts1 = [1, 2]  # 每个sample在最后一个维度上的右移量
    dim1 = 3  # 在最后一个维度上进行移位
    result1 = roll_tensor(tensor1, shifts1, dim1)
    print("示例 1 - 在最后一个维度上移位:")
    print(result1)
    
    # 示例用法 2
    tensor2 = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                            [[7, 8], [9, 10], [11, 12]],
                            [[13, 14], [15, 16], [17, 18]]])
    shifts2 = [1, 2, 3]  # 每个sample在第二个维度上的左移量
    dim2 = 1  # 在第二个维度上进行移位
    result2 = roll_tensor(tensor2, shifts2, dim2)
    print("示例 2 - 在第二个维度上移位:")
    print(result2)