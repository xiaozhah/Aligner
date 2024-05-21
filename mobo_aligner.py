import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import monotonic_align
from layers import LinearNorm
from tensor_utils import *


class MoBoAligner(nn.Module):
    def __init__(
        self, text_channels, mel_channels, attention_dim, noise_scale=2.0, max_dur=10
    ):
        super(MoBoAligner, self).__init__()
        self.mel_layer = LinearNorm(
            mel_channels, attention_dim, bias=True, w_init_gain="tanh"
        )
        self.text_layer = LinearNorm(
            text_channels, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(
            attention_dim,
            1,
            bias=True,
            w_init_gain="relu",  # because the SoftPlus is a smooth approximation to the ReLU function
            weight_norm=True,
            init_weight_norm=math.sqrt(1 / attention_dim),
        )

        self.noise_scale = noise_scale
        self.max_dur = max_dur  # max duration of a text token

    def check_parameter_validity(self, text_mask, mel_mask, direction):
        """
        Check if the parameters are valid for the alignment.
        """
        I = text_mask.sum(1)
        J = mel_mask.sum(1)
        if not (
            len(direction) >= 1
            and set(direction).issubset(set(["forward", "backward"]))
        ):
            raise ValueError(
                f"Direction must be a subset of 'forward' or 'backward', {direction} is not allowed."
            )
        if not torch.all(I < J):
            raise ValueError(
                f"The length of text hiddens is greater than or equal to the length of mel hiddens, which is not allowed."
            )
        if not torch.all(I * self.max_dur >= J):
            raise ValueError(
                f"The length of mel hiddens is greater than or equal to the {self.max_dur} times of the length of text hiddens, which is not allowed. Try to increase the max_dur or use RoMoAligner."
            )

    def compute_energy(self, text_hiddens, mel_hiddens, alignment_mask):
        """
        Compute the energy matrix between text hiddens and mel hiddens, which must be contain positional information.

        Args:
            text_hiddens (torch.FloatTensor): The text hiddens of shape (B, I, D_text).
            mel_hiddens (torch.FloatTensor): The mel hiddens of shape (B, J, D_mel).
            alignment_mask (torch.BoolTensor): The alignment mask of shape (B, I, J).

        Returns:
            energy (torch.FloatTensor): The energy matrix of shape (B, I, J) which applied Gaussian noise.
        """
        processed_mel = self.mel_layer(mel_hiddens.unsqueeze(1))  # (B, 1, J, D_att)
        processed_text = self.text_layer(text_hiddens.unsqueeze(2))  # (B, I, 1, D_att)
        energy = self.v(torch.tanh(processed_mel + processed_text))  # (B, I, J, 1)

        energy = energy.squeeze(-1)  # (B, I, J)

        noise = torch.randn_like(energy) * self.noise_scale
        # energy = F.sigmoid(energy + noise).log()
        energy = -F.softplus(-energy - noise)  # log(sigmoid(x)) = -softplus(-x)
        energy.masked_fill_(~alignment_mask, LOG_EPS)

        return energy

    def compute_reversed_energy_and_masks(self, energy, text_mask, mel_mask):
        """
        Compute the backward energy matrix and the corresponding text and mel masks.

        Args:
            energy (torch.FloatTensor): The energy matrix of shape (B, I, J).
            text_mask (torch.BoolTensor): The text mask of shape (B, I).
            mel_mask (torch.BoolTensor): The mel hidden mask of shape (B, J).

        Returns:
            energy_backward (torch.FloatTensor): The backward energy matrix of shape (B, I-1, J-1).
            text_mask_backward (torch.BoolTensor): The backward text mask of shape (B, I-1).
            mel_mask_backward (torch.BoolTensor): The backward mel hidden mask of shape (B, J-1).
        """
        shifts_text_dim = compute_max_length_diff(text_mask)
        shifts_mel_dim = compute_max_length_diff(mel_mask)

        energy_backward = shift_tensor(
            energy.flip(1, 2).unsqueeze(-1),
            shifts_text_dim=-shifts_text_dim,
            shifts_mel_dim=-shifts_mel_dim,
        ).squeeze(-1)

        energy_backward = energy_backward[:, 1:, 1:]
        text_mask_backward = text_mask[:, 1:]
        mel_mask_backward = mel_mask[:, 1:]
        alignment_mask_backward = compute_alignment_mask(
            text_mask_backward, mel_mask_backward
        )  # (B, I-1, J-1)
        return (
            energy_backward,
            text_mask_backward,
            mel_mask_backward,
            alignment_mask_backward,
        )

    def compute_cond_prob(self, energy, text_mask, mel_mask, max_dur):
        """
        Compute the log conditional probability of the alignment in the specified direction.

        Args:
            energy (torch.FloatTensor): The energy matrix of shape (B, I, J) for forward, or (B, I-1, J-1) for backward.
            text_mask (torch.BoolTensor): The text mask of shape (B, I) for forward, or (B, I-1) for backward.
            mel_mask (torch.BoolTensor): The mel hidden mask of shape (B, J) for forward, or (B, J-1) for backward.
            max_dur (int): The maximum duration of a text token.

        Returns:
            log_cond_prob (torch.FloatTensor): The log conditional probability tensor of shape (B, I, D, K) for forward, or (B, I-1, D+1, K-1) for backward.
            log_cond_prob_geq (torch.FloatTensor): The log cumulative conditional probability tensor of shape (B, I, D, K) for forward, or (B, I-1, D+1, K-1) for backward.
        """
        B, I = text_mask.shape
        _, K = mel_mask.shape  # K = J, j index from 1 to J, k index from 0 to J-1

        energy_4D = BIJ_to_BIDK(energy, max_dur, padding_direction="right")
        valid_mask = gen_left_right_mask(B, I, max_dur, K, text_mask, mel_mask)
        energy_4D.masked_fill_(~valid_mask, LOG_EPS)

        log_cond_prob = energy_4D - torch.logsumexp(
            energy_4D, dim=2, keepdim=True
        )  # on the D dimension
        log_cond_prob.masked_fill_(~valid_mask, LOG_EPS)

        log_cond_prob_geq = torch.logcumsumexp(log_cond_prob.flip(2), dim=2).flip(2)
        log_cond_prob_geq.masked_fill_(~valid_mask, LOG_EPS)

        return log_cond_prob, log_cond_prob_geq

    def compute_boundary_prob(self, log_cond_prob, text_mask, mel_mask):
        """
        Compute forward recursively in the log domain.

        Args:
            log_cond_prob (torch.FloatTensor): The log conditional probability tensor for forward of shape (B, I, D, K) for forward, or (B, I-1, D+1, K-1) for backward.
            text_mask (torch.BoolTensor): The text mask of shape (B, I) for forward, or (B, I-1) for backward.
            mel_mask (torch.BoolTensor): The mel hidden mask of shape (B, J) for forward, or (B, J-1) for backward.

        Returns:
            log_boundary_prob (torch.FloatTensor): The forward tensor of shape (B, I+1, J+1) for forward, or (B, I, J) for backward.
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        log_boundary_prob = torch.full(
            (B, I + 1, J + 1), LOG_EPS, device=log_cond_prob.device, dtype=torch.float
        )
        log_boundary_prob[:, 0, 0] = 0  # Initialize forward[0, 0] = 0
        for i in range(1, I + 1):
            log_boundary_prob[:, i, i:] = diag_logsumexp(
                log_boundary_prob[:, i - 1, :-1].unsqueeze(1)
                + log_cond_prob[:, i - 1],  # (B, D, J)
                from_ind=i - 1,
            )  # sum at the D dimension

        return log_boundary_prob

    def compute_interval_prob(
        self, boundary_prob, log_cond_prob_geq_or_gt, text_mask, alignment_mask
    ):
        """
        Compute the log interval probability, which is the log of the probability P(B_{i-1} < j <= B_i), the sum of P(B_{i-1} < j <= B_i) over i is 1.

        Args:
            boundary_prob (torch.FloatTensor): The forward or backward tensor of shape (B, I, J) for forward, or (B, I-1, J-1) for backward.
            log_cond_prob_geq_or_gt (torch.FloatTensor): The log cumulative conditional probability tensor of shape (B, I, D, K) for forward, or (B, I-1, D, J-1) for backward.
            text_mask (torch.BoolTensor): The text mask of shape (B, I) for forward, or (B, I-1) for backward.
            alignment_mask (torch.BoolTensor): The alignment mask of shape (B, I, J) for forward, or (B, I-1, J-1) for backward.

        Returns:
            log_interval_prob (torch.FloatTensor): The log interval probability tensor of shape (B, I, J) for forward, or (B, I-1, J-1) for backward.
        """
        D = log_cond_prob_geq_or_gt.shape[2]
        prob_trans = BIJ_to_BIDK(
            boundary_prob, D=D, padding_direction="left"
        )  # -> (B, I, D, K) for forward , or (B, I-1, D-1, K-1) for backward
        log_cond_prob_geq_or_gt_trans = BIDK_transform(
            log_cond_prob_geq_or_gt
        )  # -> (B, I, D, K) for forward, or (B, I-1, D-1, K-1) for backward

        log_interval_prob = torch.logsumexp(
            prob_trans[:, :-1] + log_cond_prob_geq_or_gt_trans[:, :-1], dim=2
        )  # (B, I-1, J) for forward, or (B, I-2, J-1) for backward

        log_interval_prob = force_assign_last_text_prob(
            log_interval_prob, boundary_prob, text_mask, alignment_mask
        )  # (B, I, J) for forward, or (B, I-1, J-1) for backward

        return log_interval_prob

    def combine_alignments(self, log_interval_forward, log_interval_backward):
        """
        Combine the log probabilities from forward and backward boundary calculations.

        Args:
            log_interval_forward (torch.FloatTensor): The log probabilities from the forward boundary calculation of shape (B, I, J).
            log_interval_backward (torch.FloatTensor): The log probabilities from the backward boundary calculation of shape (B, I, J).

        Returns:
            log_interval_prob (torch.FloatTensor): The combined log probabilities of shape (B, I, J).
        """
        log_interval_prob = torch.logaddexp(
            log_interval_forward - LOG_2, log_interval_backward - LOG_2
        )
        return log_interval_prob

    @torch.no_grad()
    def compute_hard_alignment(self, log_probs, alignment_mask):
        """
        Compute the Viterbi path for the maximum alignment probabilities.

        This function uses `monotonic_align.maximum_path` to find the path with the maximum probabilities,
        subject to the constraints of the alignment mask.

        Args:
            log_probs (torch.FloatTensor): The log probabilities tensor of shape (B, I, J).
            alignment_mask (torch.BoolTensor): The alignment mask of shape (B, I, J).

        Returns:
            hard_alignment (torch.FloatTensor): The tensor representing the hard alignment path of shape (B, I, J).
        """
        hard_alignment = monotonic_align.maximum_path(log_probs, alignment_mask)
        return hard_alignment

    def forward(
        self,
        text_hiddens: torch.FloatTensor,
        mel_hiddens: torch.FloatTensor,
        text_mask: torch.BoolTensor,
        mel_mask: torch.BoolTensor,
        direction: List[str],
        return_hard_alignment: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Compute the soft alignment and the expanded text hiddens.

        Args:
            text_hiddens (torch.FloatTensor): The text hiddens of shape (B, I, D_text).
            mel_hiddens (torch.FloatTensor): The mel hiddens of shape (B, J, D_mel).
            text_mask (torch.BoolTensor): The text mask of shape (B, I).
            mel_mask (torch.BoolTensor): The mel hidden mask of shape (B, J).
            direction (List[str]): The direction of the alignment, a subset of ["forward", "backward"].
            return_hard_alignment (bool): Whether to return the hard alignment which obtained by Viterbi decoding.

        Returns:
            soft_alignment (torch.FloatTensor): The soft alignment tensor of shape (B, I, J) in the log domain.
            hard_alignment (torch.FloatTensor): The hard alignment tensor of shape (B, I, J).
        """
        # Check length of text < length of mel and direction is either "forward" or "backward"
        self.check_parameter_validity(text_mask, mel_mask, direction)

        alignment_mask = compute_alignment_mask(text_mask, mel_mask)

        # Compute the energy matrix and apply noise
        energy = self.compute_energy(text_hiddens, mel_hiddens, alignment_mask)

        if "forward" in direction:
            # 1. Compute the log conditional probability P(B_i=j | B_{i-1}=k), P(B_i >= j | B_{i-1}=k) for forward
            log_cond_prob_forward, log_cond_prob_geq_forward = self.compute_cond_prob(
                energy, text_mask, mel_mask, self.max_dur
            )

            # 2. Compute forward recursively in the log domain
            log_boundary_prob_forward = self.compute_boundary_prob(
                log_cond_prob_forward, text_mask, mel_mask
            )
            log_boundary_prob_forward = BIJ_to_BIK(log_boundary_prob_forward)

            # 3. Compute the forward P(B_{i-1} < j <= B_i)
            log_interval_forward = self.compute_interval_prob(
                log_boundary_prob_forward,
                log_cond_prob_geq_forward,
                text_mask,
                alignment_mask,
            )

        if "backward" in direction:
            # 1.1 Compute the energy matrix for backward direction
            (
                energy_backward,
                text_mask_backward,
                mel_mask_backward,
                alignment_mask_backward,
            ) = self.compute_reversed_energy_and_masks(energy, text_mask, mel_mask)

            # 1.2 Compute the log conditional probability P(B_i=j | B_{i+1}=k), P(B_i < j | B_{i+1}=k) for backward
            log_cond_prob_backward, log_cond_prob_geq_backward = self.compute_cond_prob(
                energy_backward,
                text_mask_backward,
                mel_mask_backward,
                self.max_dur
                + 1,  # instead of self.max_dur, because considering the geq to gt in the next step
            )
            log_cond_prob_gt_backward = convert_geq_to_gt(log_cond_prob_geq_backward)

            # 2. Compute backward recursively in the log domain
            log_boundary_prob_backward = self.compute_boundary_prob(
                log_cond_prob_backward, text_mask_backward, mel_mask_backward
            )
            log_boundary_prob_backward = BIJ_to_BIK(log_boundary_prob_backward)

            # 3.1 Compute the backward P(B_{i-1} < j <= B_i)
            log_interval_backward = self.compute_interval_prob(
                log_boundary_prob_backward,
                log_cond_prob_gt_backward,
                text_mask_backward,
                alignment_mask_backward,
            )

            # 3.2 reverse the text and mel direction of log_interval_backward, and pad head and tail one-hot vector on mel dimension
            log_interval_backward = pad_and_reverse(
                log_interval_backward, text_mask_backward, mel_mask_backward
            )

        # Combine the forward and backward soft alignment
        if direction == ["forward"]:
            log_soft_alignment = log_interval_forward
        elif direction == ["backward"]:
            log_soft_alignment = log_interval_backward
        else:
            log_soft_alignment = self.combine_alignments(
                log_interval_forward, log_interval_backward
            )

        soft_alignment = torch.exp(log_soft_alignment) * alignment_mask

        hard_alignment = None
        if return_hard_alignment:
            hard_alignment = self.compute_hard_alignment(
                log_soft_alignment, alignment_mask
            )

        return soft_alignment, hard_alignment


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # Set a random seed to ensure reproducibility of the results
    torch.manual_seed(1234)

    I = 10
    J = 91
    device = "cpu"

    if device == "cuda":
        torch.cuda.reset_max_memory_allocated()  # 重置显存使用情况

    # Initialize the text and mel hidden tensors
    text_hiddens = torch.randn(
        2, I, 10, requires_grad=True, device=device
    )  # Batch size: 2, Text tokens: I, hidden dimension: 10
    mel_hiddens = torch.randn(
        2, J, 10, requires_grad=True, device=device
    )  # Batch size: 2, Mel frames: J, hidden dimension: 10
    # Initialize the text and mel masks
    text_mask = torch.tensor(
        [[1] * I, [1] * 10 + [0] * (I - 10)], dtype=torch.bool, device=device
    )  # Batch size: 2, Text tokens: I
    mel_mask = torch.tensor(
        [[1] * J, [1] * 70 + [0] * (J - 70)], dtype=torch.bool, device=device
    )  # Batch size: 2, Mel frames: J

    # Initialize the MoBoAligner model
    aligner = MoBoAligner(text_hiddens.size(-1), mel_hiddens.size(-1), 128).to(device)

    soft_alignment, hard_alignment = aligner(
        text_hiddens,
        mel_hiddens,
        text_mask,
        mel_mask,
        direction=["forward", "backward"],
        return_hard_alignment=True,
    )

    # Print the shape of the soft and hard alignment and the expanded text hiddens
    print("Soft alignment:")
    print(soft_alignment.shape)
    print("Hard alignment:")
    print(hard_alignment.shape)
    print("Expanded text hiddens:")
    print(soft_alignment.mean())

    if device == "cuda":
        # Print the memory usage
        print(
            f"Memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
        )

    # Backward pass test
    with torch.autograd.detect_anomaly():
        soft_alignment.mean().backward()

    print("Gradient for text_hiddens:")
    print(text_hiddens.grad.mean())
    print("Gradient for mel_hiddens:")
    print(mel_hiddens.grad.mean())
