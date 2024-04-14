from MoBoAligner import MoBoAligner
import torch

# Set a random seed to ensure reproducibility of the results
torch.manual_seed(1234)

# Initialize the text and mel embedding tensors
text_embeddings = torch.randn(2, 5, 10, requires_grad=True)  # Batch size: 2, Text tokens: 5, Embedding dimension: 10
mel_embeddings = torch.randn(2, 800, 10, requires_grad=True)  # Batch size: 2, Mel frames: 800, Embedding dimension: 10

temperature_ratio = 0.5  # Temperature ratio for Gumbel noise

# Initialize the MoBoAligner model
aligner = MoBoAligner()

gamma, expanded_text_embeddings = aligner(text_embeddings, mel_embeddings, temperature_ratio)
# gamma still in the log domain

# Print the shape of the soft alignment (gamma) and the expanded text embeddings
print("Soft alignment (gamma):")
print(gamma.shape)
print("Expanded text embeddings:")
print(expanded_text_embeddings)

# Backward pass test
gamma.sum().backward()

print("Gradient for text_embeddings:")
print(text_embeddings.grad)
print("Gradient for mel_embeddings:")
print(mel_embeddings.grad)