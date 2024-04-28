from MoBoAligner import MoBoAligner
import torch
import numpy as np

torch.autograd.set_detect_anomaly(True)

# Set a random seed to ensure reproducibility of the results
torch.manual_seed(1234)

I = 10
J = 100
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
    [[1] * I, [1] * 2 + [0] * (I - 2)], dtype=torch.bool, device=device
)  # Batch size: 2, Text tokens: I
mel_mask = torch.tensor(
    [[1] * J, [1] * 70 + [0] * (J - 70)], dtype=torch.bool, device=device
)  # Batch size: 2, Mel frames: J

# Initialize the MoBoAligner model
aligner = MoBoAligner(text_embeddings.size(-1), mel_embeddings.size(-1), 128).to(device)

soft_alignment, hard_alignment, expanded_text_embeddings = aligner(
    text_embeddings,
    mel_embeddings,
    text_mask,
    mel_mask,
    direction=["forward", "backward"],
    return_hard_alignment=True,
)

# Print the shape of the soft and hard alignment and the expanded text embeddings
print("Soft alignment:")
print(soft_alignment.shape)
print("Hard alignment:")
print(hard_alignment.shape)
print("Expanded text embeddings:")
print(expanded_text_embeddings.mean())

# Backward pass test
with torch.autograd.detect_anomaly():
    expanded_text_embeddings.mean().backward()

print("Gradient for text_embeddings:")
print(text_embeddings.grad.mean())
print("Gradient for mel_embeddings:")
print(mel_embeddings.grad.mean())
