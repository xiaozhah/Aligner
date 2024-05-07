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
    int_dur = robo_utils.float_to_int_duration_batch(dur, T, mask)

    expected_output = np.array([[1, 1, 2, 2],
                                [1, 1, 1, 2]])
    assert np.array_equal(int_dur.data.cpu().numpy(), expected_output)
    
    # Case 2
    dur = torch.FloatTensor([[0.1, 19.2, 0, 0, 0],
                             [0.2, 0.3, 0.4, 0, 0]])
    T = torch.LongTensor([2, 3])
    mask = torch.BoolTensor([[1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0]])
    int_dur = robo_utils.float_to_int_duration_batch(dur, T, mask)

    expected_output = np.array([[1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0]])
    
    print(int_dur)
    assert np.array_equal(int_dur.data.cpu().numpy(), expected_output)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_float_to_int_duration_batch()