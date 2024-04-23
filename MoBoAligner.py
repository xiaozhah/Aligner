import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mask import *
from roll import roll_tensor
import numpy as np
import warnings


class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def check_parameter_validity(self, I, J, direction):
        assert direction in ["alpha", "beta"] # direction must be "alpha" or "beta"
        if I >= J:
            warnings.warn("Warning: The dimension of text embeddings (I) is greater than or equal to the dimension of mel spectrogram embeddings (J), which may affect alignment performance.")

    def compute_energy(self, text_embeddings, mel_embeddings):
        """
        Compute the energy matrix between text embeddings and mel embeddings.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, D_text).
            mel_embeddings (torch.Tensor): The mel spectrogram embeddings of shape (B, J, D_mel).

        Returns:
            torch.Tensor: The energy matrix of shape (B, I, J).
        """
        text_channels = text_embeddings.size(-1)
        mel_channels = mel_embeddings.size(-1)
        return torch.bmm(text_embeddings, mel_embeddings.transpose(1, 2)) / math.sqrt(
            text_channels * mel_channels
        )

    def apply_gumbel_noise(self, energy, temperature_ratio):
        """
        Apply Gumbel noise and temperature to the energy matrix.

        Args:
            energy (torch.Tensor): The energy matrix of shape (B, I, J).
            temperature_ratio (float): The temperature ratio for Gumbel noise.

        Returns:
            torch.Tensor: The energy matrix with Gumbel noise and temperature applied.
        """
        temperature = (
            self.temperature_min
            + (self.temperature_max - self.temperature_min) * temperature_ratio
        )
        noise = -torch.log(-torch.log(torch.rand_like(energy)))
        return (energy + noise) / temperature

    def compute_log_cond_prob(self, energy, text_mask, mel_mask, direction="alpha"):
        """
        Compute the log conditional probability of the alignment in the specified direction.

        Args:
            energy (torch.Tensor): The energy matrix of shape (B, I, J).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).
            direction (str): The direction of the alignment, either "alpha" or "beta".

        Returns:
            tuple: A tuple containing:
                - log_cond_prob (torch.Tensor): The log conditional probability tensor of shape (B, I, J).
                - log_cond_prob_geq (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I, J)
                    for "alpha" direction, or None for "beta" direction.
                - log_cond_prob_lt (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I, J)
                    for "beta" direction, or None for "alpha" direction.
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape
        self.check_parameter_validity(I, J, direction)
        K = J if direction == "alpha" else J - 1
        
        tri = gen_tri(B, I, J, K, direction)
        ijk_mask = gen_ijk_mask(text_mask, mel_mask, direction)
        energy_mask = gen_i_range_mask(B, I, J, K, text_mask.sum(1))
        tri_ijk_mask = tri * ijk_mask
        
        ik_mask = gen_ik_mask(text_mask, mel_mask, direction)
        most_i_mask = gen_most_i_mask(B, I, J, K)

        energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, K)  # (B, I, J, K)

        energy_4D.masked_fill_(~energy_mask, -1000)
        energy_4D.masked_fill_(~tri_ijk_mask, -float("Inf"))
        energy_4D.masked_fill_(~ik_mask, -10)
        energy_4D.masked_fill_(~most_i_mask, -float("Inf"))
        log_cond_prob = energy_4D - torch.logsumexp(energy_4D, dim=2, keepdim=True) # on the J dimension
        log_cond_prob.masked_fill_(~ik_mask, -float("Inf"))

        if direction == "alpha":
            log_cond_prob_geq = torch.logcumsumexp(log_cond_prob.flip(2), dim=2).flip(2)
            log_cond_prob_geq.masked_fill_(~tri_ijk_mask, -float("Inf"))   
            return log_cond_prob, log_cond_prob_geq, None
        else:  # direction == "beta"
            log_cond_prob_lt = torch.logcumsumexp(log_cond_prob.roll(shifts=1, dims=2), dim=2)
            log_cond_prob_lt.masked_fill_(~tri_ijk_mask.roll(shifts=1, dims=2), -float("Inf"))
            return log_cond_prob, None, log_cond_prob_lt

    def right_shift(self, x, shifts_text_dim, shifts_mel_dim):
        """
        Shift the tensor x to the right along the text and mel dimensions.

        Args:
            x (torch.Tensor): The input tensor of shape (B, I, J, K).
            shifts_text_dim (torch.Tensor): The shift amounts along the text dimension of shape (B,).
            shifts_mel_dim (torch.Tensor): The shift amounts along the mel dimension of shape (B,).

        Returns:
            torch.Tensor: The right-shifted tensor of shape (B, I, J, K).
        """
        x = roll_tensor(x, shifts=shifts_text_dim, dim=1)
        x = roll_tensor(x, shifts=shifts_mel_dim, dim=2)
        shifts_mel_dim[1:] = shifts_mel_dim[1:] - 1
        x = roll_tensor(x, shifts=shifts_mel_dim, dim=3)
        return x

    def left_shift(self, x, shifts_text_dim, shifts_mel_dim):
        """
        Shift the tensor x to the left along the text and mel dimensions.

        Args:
            x (torch.Tensor): The input tensor of shape (B, I, J).
            shifts_text_dim (torch.Tensor): The shift amounts along the text dimension of shape (B,).
            shifts_mel_dim (torch.Tensor): The shift amounts along the mel dimension of shape (B,).

        Returns:
            torch.Tensor: The left-shifted tensor of shape (B, I, J).
        """
        x = x.unsqueeze(-1)
        x = roll_tensor(x, shifts=-shifts_text_dim, dim=1)
        x = roll_tensor(x, shifts=-shifts_mel_dim, dim=2)
        x = x.squeeze(-1)
        return x

    def compute_max_length_diff(self, mask):
        """
        Compute the difference between the maximum length and the actual length for each sequence in the batch.

        Args:
            mask (torch.Tensor): The mask tensor of shape (B, L).

        Returns:
            torch.Tensor: The difference tensor of shape (B,).
        """
        lengths = mask.sum(1)
        return lengths.max() - lengths

    def compute_alpha(self, log_cond_prob_alpha, text_mask, mel_mask):
        """
        Compute alpha recursively in the log domain.

        Args:
            log_cond_prob_alpha (torch.Tensor): The log conditional probability tensor for alpha of shape (B, I, J, K).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The alpha tensor of shape (B, I+1, J+1).
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        alpha = torch.full((B, I + 1, J + 1), -float("inf"), device=log_cond_prob_alpha.device)
        alpha[:, 0, 0] = 0  # Initialize alpha[0, 0] = 0
        alpha[:, -1, -1] = 0
        for i in range(1, I):
            alpha[:, i, i:] = torch.logsumexp(
                alpha[:, i - 1, :-1].unsqueeze(1)
                + log_cond_prob_alpha[:, i - 1, (i - 1) :],
                dim=2, # sum at the K dimension
            )

        return alpha

    def compute_beta(self, log_cond_prob_beta, text_mask, mel_mask):
        """
        Compute beta recursively in the log domain.

        Args:
            log_cond_prob_beta (torch.Tensor): The log conditional probability tensor for beta of shape (B, I, J, K).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The beta tensor of shape (B, I, J).
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        beta = torch.full((B, I, J), -float("inf"), device=log_cond_prob_beta.device)
        beta[:, -1, -1] = 0  # Initialize beta_{I,J} = 1
        for i in range(I - 2, -1, -1):
            beta[:, i, : (J + i - I + 1)] = torch.logsumexp(
                beta[:, i + 1, 1:].unsqueeze(1)
                + log_cond_prob_beta[:, i, : (J + i - I + 1)],
                dim=2, # sum at the K dimension
            )

        return beta
    
    def cal_delta_forward(self, alpha, log_cond_prob_alpha_geq, text_mask, mel_mask):
        B, I = text_mask.shape
        _, J = mel_mask.shape
        x = alpha[:, :-1, :-1].unsqueeze(-1).repeat(1, 1, 1, J) + log_cond_prob_alpha_geq.transpose(2, 3)
        mask = gen_upper_left_mask(B, I, J, J)
        x.masked_fill_(mask == 0, -10)
        x = torch.logsumexp(x, dim = 2)
        return x

    def forward(
        self, text_embeddings, mel_embeddings, text_mask, mel_mask, temperature_ratio
    ):
        """
        Compute the soft alignment (gamma) and the expanded text embeddings.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, D_text).
            mel_embeddings (torch.Tensor): The mel spectrogram embeddings of shape (B, J, D_mel).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).
            temperature_ratio (float): The temperature ratio for Gumbel noise.

        Returns:
            tuple: A tuple containing:
                - delta (torch.Tensor): The soft alignment tensor of shape (B, I, J) in the log domain.
                - expanded_text_embeddings (torch.Tensor): The expanded text embeddings of shape (B, J, D_text).
        """
        # Compute the energy matrix
        energy = self.compute_energy(text_embeddings, mel_embeddings)

        # Apply Gumbel noise and temperature
        energy = self.apply_gumbel_noise(energy, temperature_ratio)

        # Compute the log conditional probability P(B_i=j | B_{i-1}=k), P(B_i \geq j | B_{i-1}=k) for alpha
        log_cond_prob_alpha, log_cond_prob_alpha_geq, _ = self.compute_log_cond_prob(
            energy, text_mask, mel_mask, direction="alpha"
        )  # (B, I, J)

        # Compute the log conditional probability P(B_i=j | B_{i+1}=k), P(B_i \leq j | B_{i+1}=k) for beta
        log_cond_prob_beta, _, log_cond_prob_beta_lt = self.compute_log_cond_prob(
            energy, text_mask, mel_mask, direction="beta"
        )  # (B, I, J)
        log_cond_prob_beta = self.right_shift(
            log_cond_prob_beta,
            shifts_text_dim=self.compute_max_length_diff(text_mask),
            shifts_mel_dim=self.compute_max_length_diff(mel_mask),
        )
        log_cond_prob_beta_lt = self.right_shift(
            log_cond_prob_beta_lt,
            shifts_text_dim=self.compute_max_length_diff(text_mask),
            shifts_mel_dim=self.compute_max_length_diff(mel_mask),
        )

        # Compute alpha recursively in the log domain
        alpha = self.compute_alpha(log_cond_prob_alpha, text_mask, mel_mask)

        # Compute beta recursively in the log domain
        beta = self.compute_beta(log_cond_prob_beta, text_mask, mel_mask)
        beta = self.left_shift(
            beta,
            shifts_text_dim=self.compute_max_length_diff(text_mask),
            shifts_mel_dim=self.compute_max_length_diff(mel_mask),
        )

        # Compute the forward and backward P(B_{i-1}<j\leq B_i)
        log_delta_forward = self.cal_delta_forward(alpha, log_cond_prob_alpha_geq, text_mask, mel_mask)

        # Use log_delta to compute the expanded text embeddings
        log_delta = log_delta_forward
        expanded_text_embeddings = torch.bmm(torch.exp(log_delta).transpose(1, 2), text_embeddings)
        expanded_text_embeddings = expanded_text_embeddings * mel_mask.unsqueeze(2)

        return log_delta, expanded_text_embeddings
