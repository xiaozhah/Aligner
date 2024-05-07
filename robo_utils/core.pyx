cimport cython
from cython.parallel import prange
import numpy as np

cdef int round_to_int(float x) nogil:
    return <int>(x + 0.5)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void float_to_int_duration_each(float[:] dur, int[:] int_dur, int T, int[:] mask) nogil:
    """
    Convert float duration to int duration

    Args:
        dur (float[:]): input float duration, shape (L,)
        int_dur (int[:]): output int duration, shape (L,)
        T (int): total duration
        mask (int[:]): mask, shape (L,)
    """
    cdef int L = dur.shape[0]
    cdef float float_sum = 0
    cdef int int_sum = 0, j, rounded_dur, valid_count
    
    valid_count = 0
    for i in range(L):
        if mask[i] == 1:
            valid_count += 1
            float_sum += dur[i]
            rounded_dur = round_to_int(float_sum - int_sum)
            
            if rounded_dur <= 0:
                rounded_dur = 1  # Ensure each duration is greater than 0
            
            int_dur[i] = rounded_dur
            int_sum += rounded_dur
    
    # Adjust the last element to match the total duration
    if valid_count > 0:
        int_dur[L - 1] += T - int_sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void float_to_int_duration_batch_c(float[:, :] dur, int[:] T, int[:, :] mask, int[:, :] int_dur) nogil:
    """
    Args:
        dur (float[:, :]): float duration, shape (B, n)
        T (int[:]): total duration, shape (B,)
        mask (int[:, :]): mask, shape (B, n)
        int_dur (int[:, :]): int duration, shape (B, n)
    """
    cdef int B = dur.shape[0]
    
    cdef int i
    for i in prange(B, nogil=True):
        float_to_int_duration_each(dur[i], int_dur[i], T[i], mask[i])