import torch
import torch.nn as nn
import math
from mask import *
from roll import roll_tensor
import numpy as np
import warnings
import monotonic_align


class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.log_eps = -1000

    def check_parameter_validity(self, text_mask, mel_mask, direction):
        assert set([direction]).issubset(set(["alpha", "beta"])), "Direction must be a subset of 'alpha' or 'beta'."
        if torch.any(text_mask.sum(1) >= mel_mask.sum(1)):
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
        K = J if direction == "alpha" else J - 1
        
        energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, K)  # (B, I, J, K)
        tri_invalid = get_invalid_tri_mask(B, I, J, K, text_mask, mel_mask, direction)
        energy_4D.masked_fill_(tri_invalid, self.log_eps)
        log_cond_prob = energy_4D - torch.logsumexp(energy_4D, dim=2, keepdim=True) # on the J dimension

        if direction == "alpha":
            log_cond_prob_geq_or_gt = torch.logcumsumexp(log_cond_prob.flip(2), dim=2).flip(2)
        else:
            log_cond_prob_geq_or_gt = torch.logcumsumexp(log_cond_prob.flip(2), dim=2).flip(2)
        log_cond_prob_geq_or_gt.masked_fill_(tri_invalid, self.log_eps)
        return log_cond_prob, log_cond_prob_geq_or_gt

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

    def compute_forward_pass(self, log_cond_prob_alpha, text_mask, mel_mask):
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
        alpha[:, -1, -1] = 0  # Initialize alpha[I, J] = 0
        for i in range(1, I):
            alpha[:, i, i:(J - I + i + 2)] = torch.logsumexp(
                alpha[:, i - 1, :-1].unsqueeze(1)
                + log_cond_prob_alpha[:, i - 1, (i - 1) : (J - I + i + 1)],
                dim=2, # sum at the K dimension
            )

        return alpha

    def compute_boundary_prob(self, prob, log_cond_prob_geq_or_gt, text_mask, mel_mask):
        """
        Compute the log-delta, which is the log of the probability P(B_{i-1} < j <= B_i).

        Args:
            prob (torch.Tensor): The alpha or beta tensor of shape (B, I+1, J+1) for alpha, or (B, I, J) for beta.
            log_cond_prob_geq_or_gt (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I, J).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The log-delta tensor of shape (B, I, J).
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape
        x = prob[:, :-1, :-1].unsqueeze(-1).repeat(1, 1, 1, J) + log_cond_prob_geq_or_gt.transpose(2, 3) # (B, I, K, J)
        mask = gen_upper_left_mask(B, I, J, J)
        x.masked_fill_(mask == 0, self.log_eps) # for avoid logsumexp to produce -inf
        x = torch.logsumexp(x, dim = 2)
        mask = phone_boundary_mask(text_mask, mel_mask)
        y = x.masked_fill(mask, -float("Inf"))
        return y
    
    @torch.no_grad()
    def hard_alignment(self, log_probs, text_mask, mel_mask):
        """
        Compute the Viterbi path for the maximum alignment probabilities.

        Args:
            log_probs (torch.Tensor): The log probabilities tensor of shape (B, I, J).
            text_mask (torch.Tensor): The mask tensor of shape (B, I) indicating valid positions of text.
            mel_mask (torch.Tensor): The mask tensor of shape (B, J) indicating valid positions of mel.
        Returns:
            torch.Tensor: The tensor representing the hard alignment path of shape (B, I, J).
        """
        mask = text_mask.unsqueeze(-1) * mel_mask.unsqueeze(1)
        attn = monotonic_align.maximum_path(log_probs, mask)
        return attn

    def forward(
        self, text_embeddings, mel_embeddings, text_mask, mel_mask, temperature_ratio, direction
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
        # Check length of text < length of mel and direction is either "alpha" or "beta"
        self.check_parameter_validity(text_mask, mel_mask, direction)

        # Compute the energy matrix
        energy = self.compute_energy(text_embeddings, mel_embeddings)

        # Apply Gumbel noise and temperature
        energy = self.apply_gumbel_noise(energy, temperature_ratio)

        # Compute the log conditional probability P(B_i=j | B_{i-1}=k), P(B_i \geq j | B_{i-1}=k) for alpha
        log_cond_prob_alpha, log_cond_prob_alpha_geq_or_gt = self.compute_log_cond_prob(
            energy, text_mask, mel_mask, direction="alpha"
        )  # (B, I, J)
        
        # Compute the log conditional probability P(B_i=j | B_{i+1}=k), P(B_i \lt j | B_{i+1}=k) for beta
        # log_cond_prob_beta, log_cond_prob_beta_geq_or_gt = self.compute_log_cond_prob(
        #     energy, text_mask, mel_mask, direction="beta"
        # )  # (B, I, J)

        # Compute alpha and beta recursively in the log domain
        alpha = self.compute_forward_pass(log_cond_prob_alpha, text_mask, mel_mask)
        # beta = self.compute_forward_pass(log_cond_prob_beta, text_mask, mel_mask)

        # Compute the forward and backward P(B_{i-1}<j\leq B_i)
        log_delta_forward = self.compute_boundary_prob(alpha, log_cond_prob_alpha_geq_or_gt, text_mask, mel_mask)
        # log_delta_backward = self.compute_boundary_prob(beta, log_cond_prob_beta_geq_or_gt, text_mask, mel_mask, direction="beta")

        # Combine the forward and backward log-delta
        soft_alignment = log_delta_forward
        # Use soft_alignment to compute the expanded text embeddings
        expanded_text_embeddings = torch.bmm(torch.exp(soft_alignment).transpose(1, 2), text_embeddings)
        expanded_text_embeddings = expanded_text_embeddings * mel_mask.unsqueeze(2)

        hard_alignment = self.hard_alignment(soft_alignment, text_mask, mel_mask)

        return soft_alignment, hard_alignment, expanded_text_embeddings