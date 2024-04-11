import monotonic_align
import torch
import numpy as np

logp = torch.from_numpy(np.load('neg_cent.npy')).float().requires_grad_()
random_bool = torch.from_numpy(np.load('attn_mask.npy')).bool()

print(logp.shape, logp.requires_grad)
print(random_bool.shape, random_bool.requires_grad)

with torch.no_grad():
    attn = monotonic_align.maximum_path(logp, random_bool)
    print(attn.requires_grad)

print("Hard alignment (Viterbi decoding):")
print(attn.shape)