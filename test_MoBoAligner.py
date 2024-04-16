from MoBoAligner import MoBoAligner
import torch

torch.autograd.set_detect_anomaly(True)

# Set a random seed to ensure reproducibility of the results
torch.manual_seed(1234)

# Initialize the text and mel embedding tensors
text_embeddings = torch.randn(
    2, 5, 10, requires_grad=True
)  # Batch size: 2, Text tokens: 5, Embedding dimension: 10
mel_embeddings = torch.randn(
    2, 800, 10, requires_grad=True
)  # Batch size: 2, Mel frames: 800, Embedding dimension: 10
# Initialize the text and mel masks
text_mask = torch.tensor(
    [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
)  # Batch size: 2, Text tokens: 5
mel_mask = torch.tensor(
    [[1] * 800, [1] * 600 + [0] * 200], dtype=torch.bool
)  # Batch size: 2, Mel frames: 800

temperature_ratio = 0.5  # Temperature ratio for Gumbel noise

# Initialize the MoBoAligner model
aligner = MoBoAligner()

gamma, expanded_text_embeddings = aligner(
    text_embeddings, mel_embeddings, text_mask, mel_mask, temperature_ratio
)
# gamma still in the log domain

# Print the shape of the soft alignment (gamma) and the expanded text embeddings
print("Soft alignment (gamma):")
print(gamma.shape)
print("Expanded text embeddings:")
print(expanded_text_embeddings)

# Backward pass test

with torch.autograd.detect_anomaly():
    print(expanded_text_embeddings.mean())
    expanded_text_embeddings.mean().backward()

print("Gradient for text_embeddings:")
print(text_embeddings.grad)
print("Gradient for mel_embeddings:")
print(mel_embeddings.grad)
