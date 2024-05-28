cimport cython
from cython.parallel import prange
import numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor

cdef int round_to_int(float x) nogil:
    return <int>(x + 0.5)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void float_to_int_duration_each(float[:] dur, int[:] int_dur, int T, int[:] mask) nogil:
    """
    Convert float duration to int duration

    Args:
        dur (float[:]): input float duration, shape (I,)
        int_dur (int[:]): output int duration, shape (I,)
        T (int): total duration
        mask (int[:]): mask, shape (I,)
    """
    cdef int I = dur.shape[0]
    cdef float float_sum = 0
    cdef int int_sum = 0, i, rounded_dur, valid_count
    
    valid_count = 0
    for i in range(I):
        if mask[i] == 1:
            valid_count += 1
            float_sum += dur[i]
            rounded_dur = round_to_int(float_sum - int_sum)
            
            if rounded_dur <= 0:
                rounded_dur = 1  # Ensure each duration is greater than 0
            
            int_dur[i] = rounded_dur
            int_sum += rounded_dur
        else:
            break
    
    # Adjust the last valid element, so int_dur matches the total duration
    if valid_count > 0:
        int_dur[valid_count - 1] += T - int_sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void float_to_int_duration_batch_c(float[:, :] dur, int[:] T, int[:, :] mask, int[:, :] int_dur) nogil:
    """
    Args:
        dur (float[:, :]): float duration, shape (B, I)
        T (int[:]): total duration, shape (B,)
        mask (int[:, :]): mask, shape (B, I)
        int_dur (int[:, :]): int duration, shape (B, I)
    """
    cdef int B = dur.shape[0]
    
    cdef int i
    for i in prange(B, nogil=True):
        float_to_int_duration_each(dur[i], int_dur[i], T[i], mask[i])

cdef float generate_random(float start, float end) nogil:
    return start + (end - start) * (rand() / float(RAND_MAX))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void generate_random_intervals_each(float[:] boundaries, float[:] result, int num_randoms) nogil:
    """
    Generate random intervals for a single batch

    Args:
        boundaries (float[:]): input boundaries, shape (N,)
        result (float[:]): output random values, shape ((N-1) * num_randoms,)
        num_randoms (int): number of random values to generate per interval
    """
    cdef int N = boundaries.shape[0]
    cdef int idx = 0
    cdef int i, j

    for i in range(N - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        for j in range(num_randoms):
            result[idx + j] = generate_random(start, end)
        result[idx + num_randoms] = end
        idx += (num_randoms + 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void generate_random_intervals_batch_c(float[:, :] boundaries_batch, float[:, :] result_batch, int num_randoms) nogil:
    """
    Generate random intervals for a batch of boundaries

    Args:
        boundaries_batch (float[:, :]): input boundaries, shape (B, N)
        result_batch (float[:, :]): output random values, shape (B, (N-1) * num_randoms)
        num_randoms (int): number of random values to generate per interval
    """
    cdef int B = boundaries_batch.shape[0]

    cdef int i
    for i in prange(B, nogil=True):
        generate_random_intervals_each(boundaries_batch[i], result_batch[i], num_randoms)