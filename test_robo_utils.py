import robo_utils
import numpy as np
import torch

def test_float_to_int_duration_batch():
    dur = torch.FloatTensor([[0.5, 1.2, 2.7, 1.8],
                             [0.3, 0.8, 1.5, 2.4]])
    T = torch.LongTensor([6, 5])
    mask = torch.BoolTensor([[1, 1, 1, 1],
                             [1, 1, 1, 1]])
    int_dur = robo_utils.float_to_int_duration_batch(dur, T, mask)

    expected_output = np.array([[1, 1, 2, 2],
                                [1, 1, 1, 2]])
    assert np.array_equal(int_dur.data.cpu().numpy(), expected_output)
    
    # dur = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
    #                 [0.2, 0.3, 0.4, 0.5, 0.6]])
    # T = np.array([2, 3]).astype(dtype=np.int32)
    # mask = np.array([[1, 1, 1, 0, 0],
    #                  [1, 1, 1, 1, 0]]).astype(dtype=np.int32)
    # int_dur = np.zeros_like(dur, dtype=np.int32).astype(dtype=np.int32)

    # expected_output = np.array([[1, 1, 0, 0, 0],
    #                             [1, 1, 1, 0, 0]])
    
    # float_to_int_duration_batch(dur, T, mask, int_dur)
    # print(int_dur)
    # assert np.array_equal(int_dur, expected_output)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_float_to_int_duration_batch()
