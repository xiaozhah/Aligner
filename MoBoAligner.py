import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def energy_4D(self, energy, text_mask, mel_mask, direction='alpha'):
        """
        Compute the log conditional probability of the alignment in the forward or backward direction.
        """
        batch_size, max_text_length = text_mask.shape
        batch_size, max_mel_length = mel_mask.shape

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

        mask = text_mask.unsqueeze(2).unsqueeze(3) * mel_mask.unsqueeze(1).unsqueeze(3) * mel_mask.unsqueeze(1).unsqueeze(1)
        if direction == 'beta': # because K is max_mel_length-1
            mask = mask[:, :, :, :-1]
        mask = triu * mask

        energy_4D.masked_fill_(mask == 0, -float("Inf"))
        energy_4D = energy_4D - torch.logsumexp(energy_4D, dim=2, keepdim=True)
        energy_4D.masked_fill_(mask == 0, -10)
        return energy_4D

    def forward(self, text_embeddings, mel_embeddings, text_mask, mel_mask, temperature_ratio):
        """
        Compute the soft alignment (gamma) and the expanded text embeddings.
        """
        batch_size, max_text_length, text_channels = text_embeddings.size()
        _, max_mel_length, mel_channels = mel_embeddings.size()

        # Compute the energy matrix
        energy = torch.bmm(text_embeddings, mel_embeddings.transpose(1, 2)) / math.sqrt(text_channels * mel_channels)

        # Apply Gumbel noise and temperature
        temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * temperature_ratio
        noise = -torch.log(-torch.log(torch.rand_like(energy)))
        energy = (energy + noise) / temperature

        # Compute the log conditional probability P(B_i=j | B_{i-1}=k) for alpha
        cond_prob_alpha = self.energy_4D(energy, text_mask, mel_mask, direction='alpha')  # (B, I, J, K)

        # Compute the log conditional probability P(B_i=j | B_{i+1}=k) for beta
        cond_prob_beta = self.energy_4D(energy, text_mask, mel_mask, direction='beta')  # (B, I, J, K)

        # Compute alpha recursively, in the log domain
        alpha = torch.full((batch_size, max_text_length+1, max_mel_length+1), -float('inf'), device=energy.device)
        alpha[:, 0, 0] = 0  # Initialize alpha[0, 0] = 0
        for i in range(1, max_text_length+1):
            alpha[:, i, 1:] = torch.logsumexp(alpha[:, i-1, :-1].unsqueeze(1) + cond_prob_alpha[:, i-1, :], dim=2)

        # Compute beta recursively
        beta = torch.full((batch_size, max_text_length, max_mel_length), -float('inf'), device=energy.device)
        beta[:, -1, -1] = 0  # Initialize beta_{I,J} = 1
        for i in range(max_text_length-2, -1, -1):
            beta[:, i, :] = torch.logsumexp(beta[:, i+1, 1:].unsqueeze(1) + cond_prob_beta[:, i, :], dim=2)

        # Compute gamma (soft alignment)
        gamma = alpha[:, 1:, 1:] + beta

        gamma_mask = text_mask.unsqueeze(2) * mel_mask.unsqueeze(1)
        gamma.masked_fill_(gamma_mask == 0, -float("Inf"))
        gamma = gamma - torch.logsumexp(gamma, dim=1, keepdim=True) 
        gamma.masked_fill_(gamma_mask == 0, -float("Inf"))

        # Compute the expanded text embeddings
        expanded_text_embeddings = torch.bmm(torch.exp(gamma).transpose(1, 2), text_embeddings)
        expanded_text_embeddings = expanded_text_embeddings * mel_mask.unsqueeze(2)

        return gamma, expanded_text_embeddings # gamma still in the log domain