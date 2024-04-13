from MoBoAligner import MoBoAligner
import torch

# Example usage
torch.manual_seed(1234)
text_embeddings = torch.randn(2, 5, 10)  # Batch size 2, 5 text tokens, embedding dim 10
mel_embeddings = torch.randn(2, 800, 10)   # Batch size 2, 8 mel frames, embedding dim 10
temperature_ratio = 0.5                  # Temperature ratio for Gumbel noise

aligner = MoBoAligner()
gamma_log, expanded_text_embeddings = aligner(text_embeddings, mel_embeddings, temperature_ratio)

print("Soft alignment (gamma_log):")
print(gamma_log.shape)
print("Expanded text embeddings:")
print(expanded_text_embeddings)