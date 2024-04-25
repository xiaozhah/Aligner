from MoBoAligner import MoBoAligner
import torch
import numpy as np
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# Set a random seed to ensure reproducibility of the results
torch.manual_seed(1234)

for J in tqdm(range(31, 300)):
    I = 30
    device = "cpu"
    # Initialize the text and mel embedding tensors
    text_embeddings = torch.randn(
        2, I, 10, requires_grad=True, device=device
    )  # Batch size: 2, Text tokens: I, Embedding dimension: 10
    mel_embeddings = torch.randn(
        2, J, 10, requires_grad=True, device=device
    )  # Batch size: 2, Mel frames: J, Embedding dimension: 10
    # Initialize the text and mel masks
    text_mask = torch.tensor(
        [[1] * I, [1] * I], dtype=torch.bool, device=device
    )  # Batch size: 2, Text tokens: I
    mel_mask = torch.tensor(
        [[1] * J, [1] * J], dtype=torch.bool, device=device
    )  # Batch size: 2, Mel frames: J

    # Initialize the MoBoAligner model
    aligner = MoBoAligner(text_embeddings.size(-1), mel_embeddings.size(-1), 128)

    soft_alignment, hard_alignment, expanded_text_embeddings = aligner(
        text_embeddings,
        mel_embeddings,
        text_mask,
        mel_mask,
        direction=["forward"],
    )

    assert((soft_alignment.exp().sum(1).log() < 1e-4).all())

