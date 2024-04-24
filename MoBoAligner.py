from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import math
from mask_utils import get_invalid_tri_mask, get_j_last
from tensor_utils import roll_tensor_1d, right_shift, left_shift
import numpy as np
import warnings
import monotonic_align

LOG_EPS = -1000
LOG_2 = math.log(2.0)


class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def check_parameter_validity(self, text_mask, mel_mask, direction):
        assert len(direction) >= 1 and set(direction).issubset(
            set(["forward", "backward"])
        ), "Direction must be a subset of 'forward' or 'backward'."
        if torch.any(text_mask.sum(1) >= mel_mask.sum(1)):
            warnings.warn(
                "Warning: The dimension of text embeddings (I) is greater than or equal to the dimension of mel spectrogram embeddings (J), which may affect alignment performance."
            )

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

    def apply_gumbel_noise(self, energy, gumbel_temperature_ratio):
        """
        Apply Gumbel noise and temperature to the energy matrix.

        Args:
            energy (torch.Tensor): The energy matrix of shape (B, I, J).
            gumbel_temperature_ratio (float): The temperature ratio for Gumbel noise.

        Returns:
            torch.Tensor: The energy matrix with Gumbel noise and temperature applied.
        """
        temperature = (
            self.temperature_min
            + (self.temperature_max - self.temperature_min) * gumbel_temperature_ratio
        )
        noise = -torch.log(-torch.log(torch.rand_like(energy)))
        return (energy + noise) / temperature

    def compute_backward_energy_and_masks(self, energy, text_mask, mel_mask):
        shifts_text_dim = self.compute_max_length_diff(text_mask)
        shifts_mel_dim = self.compute_max_length_diff(mel_mask)

        energy_backward = left_shift(
            energy.flip(1, 2),
            shifts_text_dim=shifts_text_dim,
            shifts_mel_dim=shifts_mel_dim,
        )
        text_mask_backward = roll_tensor_1d(text_mask.flip(1), shifts=shifts_text_dim)
        mel_mask_backward = roll_tensor_1d(mel_mask.flip(1), shifts=shifts_mel_dim)

        energy_backward = energy_backward[:, 1:, 1:]
        text_mask_backward = text_mask_backward[:, 1:]
        mel_mask_backward = mel_mask_backward[:, 1:]
        return energy_backward, text_mask_backward, mel_mask_backward

    def compute_log_cond_prob(self, energy, text_mask, mel_mask, force_assign_last):
        """
        Compute the log conditional probability of the alignment in the specified direction.

        Args:
            energy (torch.Tensor): The energy matrix of shape (B, I, J).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).
            force_assign_last (bool): Whether to force the last element of mask to be assigned.

        Returns:
            tuple: A tuple containing:
                - log_cond_prob (torch.Tensor): The log conditional probability tensor of shape (B, I, J).
                - log_cond_prob_geq (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I, J)
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, J)  # (B, I, J, K)
        tri_invalid = get_invalid_tri_mask(
            B, I, J, J, text_mask, mel_mask, force_assign_last
        )
        energy_4D.masked_fill_(tri_invalid, LOG_EPS)
        log_cond_prob = energy_4D - torch.logsumexp(
            energy_4D, dim=2, keepdim=True
        )  # on the J dimension

        log_cond_prob_geq = torch.logcumsumexp(log_cond_prob.flip(2), dim=2).flip(2)
        log_cond_prob_geq.masked_fill_(tri_invalid, LOG_EPS)
        return log_cond_prob, log_cond_prob_geq

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

    def compute_forward_pass(self, log_cond_prob, text_mask, mel_mask):
        """
        Compute forward recursively in the log domain.

        Args:
            log_cond_prob (torch.Tensor): The log conditional probability tensor for forward of shape (B, I, J, K).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The forward tensor of shape (B, I+1, J+1).
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        B_ij = torch.full((B, I + 1, J + 1), -float("inf"), device=log_cond_prob.device)
        B_ij[:, 0, 0] = 0  # Initialize forward[0, 0] = 0
        for i in range(1, I + 1):
            B_ij[:, i, i : (J - I + i + 2)] = torch.logsumexp(
                B_ij[:, i - 1, :-1].unsqueeze(1)
                + log_cond_prob[:, i - 1, (i - 1) : (J - I + i + 1)],
                dim=2,  # sum at the K dimension
            )

        return B_ij

    def compute_interval_probability(self, prob, log_cond_prob_geq_or_gt, mel_mask):
        """
        Compute the log interval probability, which is the log of the probability P(B_{i-1} < j <= B_i), the sum of P(B_{i-1} < j <= B_i) over i is 1.

        Args:
            prob (torch.Tensor): The forward or backward tensor of shape (B, I, J) for forward, or (B, I, J) for backward.
            log_cond_prob_geq_or_gt (torch.Tensor): The log cumulative conditional probability tensor of shape (B, I, J).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The log interval probability tensor of shape (B, I, J).
        """
        _, J = mel_mask.shape
        x = prob.unsqueeze(-1).repeat(1, 1, 1, J) + log_cond_prob_geq_or_gt.transpose(
            2, 3
        )  # (B, I, K, J)
        x = torch.logsumexp(x, dim=2)
        return x

    def combine_alignments(self, log_boundary_forward, log_boundary_backward):
        """
        Combine the log probabilities from forward and backward boundary calculations.

        Args:
            log_boundary_forward (torch.Tensor): The log probabilities from the forward boundary calculation of shape (B, I, J).
            log_boundary_backward (torch.Tensor): The log probabilities from the backward boundary calculation of shape (B, I, J).

        Returns:
            torch.Tensor: The combined log probabilities of shape (B, I, J).
        """
        log_interval_probability = torch.logaddexp(
            log_boundary_forward - LOG_2, log_boundary_backward - LOG_2
        )
        return log_interval_probability

    @torch.no_grad()
    def compute_hard_alignment(self, log_probs, text_mask, mel_mask):
        """
        Compute the Viterbi path for the maximum alignment probabilities.

        This function uses `monotonic_align.maximum_path` to find the path with the maximum probabilities,
        subject to the constraints of the text and mel masks.

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
        self,
        text_embeddings: torch.FloatTensor,
        mel_embeddings: torch.FloatTensor,
        text_mask: torch.BoolTensor,
        mel_mask: torch.BoolTensor,
        gumbel_temperature_ratio: float,
        direction: List[str],
    ) -> Tuple[
        Optional[torch.FloatTensor], Optional[torch.FloatTensor], torch.FloatTensor
    ]:
        """
        Compute the soft alignment and the expanded text embeddings.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, D_text).
            mel_embeddings (torch.Tensor): The mel spectrogram embeddings of shape (B, J, D_mel).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).
            gumbel_temperature_ratio (float): The temperature ratio for Gumbel noise.
            direction (List[str]): The direction of the alignment, a subset of ["forward", "backward"].

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
                - soft_alignment (torch.Tensor): The soft alignment tensor of shape (B, I, J) in the log domain.
                - hard_alignment (torch.Tensor): The hard alignment tensor of shape (B, I, J).
                - expanded_text_embeddings (torch.Tensor): The expanded text embeddings of shape (B, J, D_text).
        """
        # Check length of text < length of mel and direction is either "forward" or "backward"
        self.check_parameter_validity(text_mask, mel_mask, direction)

        # Compute the energy matrix
        energy = self.compute_energy(text_embeddings, mel_embeddings)

        # Apply Gumbel noise and temperature
        energy = self.apply_gumbel_noise(energy, gumbel_temperature_ratio)

        if "forward" in direction:
            # Compute the log conditional probability P(B_i=j | B_{i-1}=k), P(B_i >= j | B_{i-1}=k) for forward
            log_cond_prob_forward, log_cond_prob_forward_geq = (
                self.compute_log_cond_prob(
                    energy, text_mask, mel_mask, force_assign_last=True
                )
            )

            # Compute forward recursively in the log domain
            Bij_forward = self.compute_forward_pass(
                log_cond_prob_forward, text_mask, mel_mask
            )
            Bij_forward = Bij_forward[:, :-1, :-1]

            # Compute the forward P(B_{i-1}<j\leq B_i)
            log_boundary_forward = self.compute_interval_probability(
                Bij_forward, log_cond_prob_forward_geq, mel_mask
            )

        if "backward" in direction:
            # Compute the energy matrix for backward direction
            energy_backward, text_mask_backward, mel_mask_backward = (
                self.compute_backward_energy_and_masks(energy, text_mask, mel_mask)
            )

            # Compute the log conditional probability P(B_i=j | B_{i+1}=k), P(B_i < j | B_{i+1}=k) for backward
            log_cond_prob_backward, log_cond_prob_geq_backward = (
                self.compute_log_cond_prob(
                    energy_backward,
                    text_mask_backward,
                    mel_mask_backward,
                    force_assign_last=False,
                )
            )

            # Compute the log conditional probability P(B_i < j | B_{i+1}=k) based on P(B_i <= j | B_{i+1}=k)
            log_cond_prob_gt_backward = log_cond_prob_geq_backward.roll(
                shifts=-1, dims=2
            )
            log_cond_prob_gt_backward_mask = get_j_last(
                log_cond_prob_gt_backward, device=text_mask.device
            )
            log_cond_prob_gt_backward.masked_fill_(
                log_cond_prob_gt_backward_mask, LOG_EPS
            )

            # Compute backward recursively in the log domain
            Bij_backward = self.compute_forward_pass(
                log_cond_prob_backward, text_mask_backward, mel_mask_backward
            )
            Bij_backward = Bij_backward[:, :-1, :-1]

            # Compute the backward P(B_{i-1}<j\leq B_i)
            log_boundary_backward = self.compute_interval_probability(
                Bij_backward, log_cond_prob_gt_backward, mel_mask_backward
            )
            shifts_text_dim = self.compute_max_length_diff(text_mask_backward)
            shifts_mel_dim = self.compute_max_length_diff(mel_mask_backward)
            log_boundary_backward = right_shift(
                log_boundary_backward.flip(1, 2),
                shifts_text_dim=shifts_text_dim,
                shifts_mel_dim=shifts_mel_dim,
            )

        # Combine the forward and backward soft alignment
        if direction == ["forward"]:
            soft_alignment = log_boundary_forward
        elif direction == ["backward"]:
            soft_alignment = log_boundary_backward
        else:
            soft_alignment = self.combine_alignments(
                log_boundary_forward, log_boundary_backward
            )

        # Use soft_alignment to compute the expanded text embeddings
        expanded_text_embeddings = torch.bmm(
            torch.exp(soft_alignment).transpose(1, 2), text_embeddings
        )
        expanded_text_embeddings = expanded_text_embeddings * mel_mask.unsqueeze(2)

        hard_alignment = self.compute_hard_alignment(
            soft_alignment, text_mask, mel_mask
        )

        return soft_alignment, hard_alignment, expanded_text_embeddings
