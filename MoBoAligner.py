import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
    
    def log_energy_4D(self, energy, batch_size, max_text_length, max_mel_length, direction='alpha'):
        if direction == 'alpha':
            energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, max_mel_length)  # (B, I, J, K)
            triu = torch.triu(torch.ones((max_mel_length, max_mel_length)), diagonal=0).to(energy.device)  # (K, J)
        elif direction == 'beta':
            energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, max_mel_length-1)  # (B, I, J, K)
            triu = torch.tril(torch.ones((max_mel_length-1, max_mel_length)), diagonal=0).to(energy.device)  # (K, J), K == J-1
        else:
            raise ValueError("direction must be 'alpha' or 'beta'")

        triu = triu.unsqueeze(-1).unsqueeze(0)  # (1, K, J, 1)
        triu = triu.repeat(batch_size, 1, 1, max_text_length)  # (B, K, J, I)
        triu = triu.transpose(1, 3)  # (B, I, J, K)
        
        energy_4D = energy_4D * triu
        energy_4D = energy_4D.log() - energy_4D.sum(2, keepdim=True).log()

        if direction == 'beta':
            energy_4D = torch.flip(energy_4D.transpose(2, 3), dims=(2, 3))
        
        return energy_4D
        
    def forward(self, text_embeddings, mel_embeddings, temperature_ratio):
        batch_size, max_text_length, text_channels = text_embeddings.size()
        _, max_mel_length, mel_channels = mel_embeddings.size()
        
        # Compute the energy matrix
        energy = torch.bmm(text_embeddings, mel_embeddings.transpose(1, 2)) / math.sqrt(text_channels * mel_channels)
        
        # Apply Gumbel noise and temperature
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * temperature_ratio
        noise = -torch.log(-torch.log(torch.rand_like(energy)))
        energy = torch.exp((energy + noise) / temperature)
        
        # Compute the log conditional probability P(B_i=j | B_{i-1}=k) for alpha
        log_cond_prob_alpha = self.log_energy_4D(energy, batch_size, max_text_length, max_mel_length, direction='alpha')  # (B, I, J, K)
        
        # Compute the log conditional probability P(B_i=j | B_{i+1}=k) for beta
        log_cond_prob_beta = self.log_energy_4D(energy, batch_size, max_text_length, max_mel_length, direction='beta')  # (B, I, J, K)
        
        # Compute alpha recursively, in the log domain
        alpha = torch.full((batch_size, max_text_length+1, max_mel_length+1), -float('inf'), device=energy.device)
        alpha[:, 0, 0] = 0 # Initialize alpha[0, 0] = 0
        
        for i in range(1, max_text_length+1):
            alpha[:, i, 1:] = torch.logsumexp(alpha[:, i-1, :-1].unsqueeze(1) + log_cond_prob_alpha[:, i-1, :], dim=2)
        
        # Compute beta recursively
        beta = torch.full((batch_size, max_text_length, max_mel_length), -float('inf'), device=energy.device)
        beta[:, -1, -1] = 0  # Initialize beta_{I,J} = 1
        
        for i in range(max_text_length-2, -1, -1):
            beta[:, i, :-1] = torch.logsumexp(beta[:, i+1, :].unsqueeze(1) + log_cond_prob_beta[:, i, :], dim=2)
        
        # Compute gamma (soft alignment)
        gamma = alpha[:, 1:, 1:] + beta
        gamma = gamma - torch.logsumexp(gamma, dim=1, keepdim=True)
        gamma = torch.exp(gamma)
        
        # Compute the expanded text embeddings
        expanded_text_embeddings = torch.bmm(gamma.transpose(1, 2), text_embeddings)
        
        return gamma, expanded_text_embeddings