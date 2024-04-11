import torch
import torch.nn as nn
import torch.nn.functional as F

class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        
    def forward(self, text_embeddings, mel_embeddings, temperature_ratio):
        batch_size, max_text_length, _ = text_embeddings.size()
        _, max_mel_length, _ = mel_embeddings.size()
        
        # Compute the energy matrix
        energy = torch.bmm(text_embeddings, mel_embeddings.transpose(1, 2))
        
        # Apply Gumbel noise and temperature
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * temperature_ratio
        noise = -torch.log(-torch.log(torch.rand_like(energy)))
        energy = (energy + noise) / temperature
        
        # Compute the conditional probability P(B_i=j | B_{i-1}=k)
        cond_prob = F.softmax(energy, dim=2)
        
        # Compute alpha recursively
        alpha = torch.zeros((batch_size, max_text_length+1, max_mel_length+1)).to(energy.device)
        alpha[:, 0, 0] = 1  # Initialize P(B_0 = 0) = 1
        
        for i in range(1, max_text_length+1):
            alpha[:, i, 1:] = torch.logaddexp(alpha[:, i-1, :-1].unsqueeze(2), torch.log(cond_prob[:, i-1, :]).unsqueeze(1)).sum(dim=1)
        
        # Compute beta recursively
        beta = torch.zeros((batch_size, max_text_length, max_mel_length)).to(energy.device)
        beta[:, -1, -1] = 1  # Initialize beta_{I,J} = 1
        
        for i in range(max_text_length-2, -1, -1):
            beta[:, i, :] = torch.logaddexp(beta[:, i+1, :].unsqueeze(1), torch.log(cond_prob[:, i, :]).unsqueeze(1)).sum(dim=2)
        
        # Compute gamma (soft alignment)
        gamma = alpha[:, 1:, 1:] + beta
        gamma_log = gamma - torch.logsumexp(gamma, dim=1, keepdim=True)
        
        # Compute the expanded text embeddings
        expanded_text_embeddings = torch.bmm(gamma.transpose(1, 2), text_embeddings)
        
        return gamma_log, expanded_text_embeddings