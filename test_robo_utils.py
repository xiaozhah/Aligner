import robo_utils
import numpy as np
import torch

def test_float_to_int_duration_batch():
    # Case 1
    dur = torch.FloatTensor([[0.5, 1.2, 2.7, 1.8],
                             [0.3, 0.8, 1.5, 2.4]])
    T = torch.LongTensor([6, 5])
    mask = torch.BoolTensor([[1, 1, 1, 1],
                             [1, 1, 1, 1]])
    int_dur = robo_utils.float_to_int_duration(dur, T, mask)

    expected_output = np.array([[1, 1, 2, 2],
                                [1, 1, 1, 2]])
    assert np.array_equal(int_dur.data.cpu().numpy(), expected_output)
    
    # Case 2
    dur = torch.FloatTensor([[0.1, 19.2, 0, 0, 0],
                             [0.2, 0.3, 0.4, 0, 0]])
    T = torch.LongTensor([2, 3])
    mask = torch.BoolTensor([[1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0]])
    int_dur = robo_utils.float_to_int_duration(dur, T, mask)

    expected_output = np.array([[1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0]])
    assert np.array_equal(int_dur.data.cpu().numpy(), expected_output)
    
    print("All tests passed!")

def test_generate_random_intervals_batch():

    # 定义一个包含多个批次边界值的数组
    boundaries_batch = torch.tensor([
        [1, 10, 20, 50, 70],
    ]).int()

    result_batch = robo_utils.generate_random_intervals(boundaries_batch, 1)
    print(result_batch)

if __name__ == "__main__":
    test_float_to_int_duration_batch()
    test_generate_random_intervals_batch()
