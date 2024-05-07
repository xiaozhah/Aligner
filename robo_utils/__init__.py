import numpy as np
import torch
from .robo_utils.core import float_to_int_duration_batch_c

def float_to_int_duration_batch(dur, T, mask):  
    """ Cython optimised version.
    Args:
        dur (torch.Tensor): input float duration, shape (B, L)
        T (torch.Tensor): input int duration, shape (B,)
        mask (torch.Tensor): mask, shape (B, L)
    Returns:
        torch.LongTensor: output int duration, shape (B, L)
    """
    dur = dur.data.cpu().numpy().astype(np.float32)
    T = T.data.cpu().numpy().astype(np.int32)
    mask = mask.data.cpu().numpy().astype(np.int32)

    int_dur = np.zeros_like(dur).astype(dtype=np.int32)
    float_to_int_duration_batch_c(dur, T, mask, int_dur)
    return torch.from_numpy(int_dur).long()
