from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rough_aligner import RoughAligner
from mobo_aligner import MoBoAligner
from tensor_utils import get_mat_p_f, get_nearest_boundaries
import robo_utils


class RoMoAligner(nn.Module):
    def __init__(
        self,
        text_channels,
        mel_channels,
        attention_dim,
        attention_head,
        dropout,
        noise_scale=2.0,
    ):
        super(RoMoAligner, self).__init__()

        self.rough_aligner = RoughAligner(
            text_channels, mel_channels, attention_dim, attention_head, dropout
        )
        self.mobo_aligner = MoBoAligner(
            text_channels, mel_channels, attention_dim, noise_scale
        )

    def select_mel_embeddings(
        self, mel_embeddings, selected_boundary_indices, selected_boundary_indices_mask
    ):
        """
        Selects the corresponding mel_embeddings according to the possible boundary indices predicted by the rough aligner.

        Args:
            mel_embeddings (torch.Tensor): The original mel feature sequence, with a shape of (B, J, C).
            selected_boundary_indices (torch.Tensor): The indices near the boundaries predicted by the rough aligner, with a shape of (B, K).

        Returns:
            torch.Tensor: The selected mel feature sequence, with a shape of (B, K, C).
        """
        channels = mel_embeddings.shape[2]

        selected_mel_embeddings = torch.gather(
            mel_embeddings,
            1,
            selected_boundary_indices.unsqueeze(-1).expand(-1, -1, channels),
        )

        selected_mel_embeddings = (
            selected_mel_embeddings * selected_boundary_indices_mask.unsqueeze(-1)
        )

        return selected_mel_embeddings

    def get_possible_boundaries(self, durations_normalized, text_mask, mel_mask, D=3):
        # Calculate the possible boundaries of each text token based on the results of the rough aligner
        # if the length of text tokens is I, the number of possible boundaries is about I*(2*D+1)
        T = mel_mask.sum(dim=1)
        float_dur = durations_normalized * T.unsqueeze(1)
        int_dur = robo_utils.float_to_int_duration(float_dur, T, text_mask)
        selected_boundary_indices, selected_boundary_indices_mask = (
            get_nearest_boundaries(int_dur, text_mask, D)
        )
        return selected_boundary_indices, selected_boundary_indices_mask

    def get_map_d_f(
        self, mat_p_d, selected_boundary_indices, selected_boundary_indices_mask
    ):
        repeat_times = F.pad(
            selected_boundary_indices, (1, 0), mode="constant", value=-1
        ).diff(1)
        repeat_times = repeat_times * selected_boundary_indices_mask
        map_d_f = get_mat_p_f(mat_p_d.transpose(1, 2), repeat_times)
        return map_d_f

    def forward(
        self,
        text_embeddings: torch.FloatTensor,
        mel_embeddings: torch.FloatTensor,
        text_mask: torch.BoolTensor,
        mel_mask: torch.BoolTensor,
        direction: List[str],
        D: int = 3,
    ) -> Tuple[
        Optional[torch.FloatTensor], Optional[torch.FloatTensor], torch.FloatTensor
    ]:
        """
        Args:
            text_embeddings (torch.FloatTensor): The input text embeddings, with a shape of (B, I, C1).
            mel_embeddings (torch.FloatTensor): The input mel embeddings, with a shape of (B, J, C2).
            text_mask (torch.BoolTensor): The mask for the input text, with a shape of (B, I).
            mel_mask (torch.BoolTensor): The mask for the input mel, with a shape of (B, J).
            direction (List[str]): The direction of the alignment, can be "forward" or "backward".
            D (int): The number of possible nearest boundary indices for each rough boundary.
        Returns:
            torch.FloatTensor: The soft alignment matrix, with a shape of (B, I, J).
            torch.FloatTensor: The hard alignment matrix, with a shape of (B, I, J).
            torch.FloatTensor: The expanded text embeddings, with a shape of (B, J, C1).
        """
        durations_normalized = self.rough_aligner(
            text_embeddings, mel_embeddings, text_mask, mel_mask
        )

        selected_boundary_indices, selected_boundary_indices_mask = (
            self.get_possible_boundaries(durations_normalized, text_mask, mel_mask, D=3)
        )

        # Select the corresponding mel_embeddings based on the possible boundary indices
        selected_mel_embeddings = self.select_mel_embeddings(
            mel_embeddings, selected_boundary_indices, selected_boundary_indices_mask
        )

        # Run a fine-grained MoBoAligner
        mat_p_d, hard_mat_p_d = self.mobo_aligner(
            text_embeddings,
            selected_mel_embeddings,
            text_mask,
            selected_boundary_indices_mask,
            direction,
            return_hard_alignment=True,
        )

        map_d_f = self.get_map_d_f(
            mat_p_d, selected_boundary_indices, selected_boundary_indices_mask
        )
        mat_p_f = torch.bmm(mat_p_d, map_d_f)
        hard_mat_p_f = torch.bmm(hard_mat_p_d, map_d_f)

        # Use mat_p_f to compute the expanded text_embeddings
        expanded_text_embeddings = torch.bmm(
            torch.exp(mat_p_f).transpose(1, 2), text_embeddings
        )
        expanded_text_embeddings = expanded_text_embeddings * mel_mask.unsqueeze(2)

        return mat_p_f, hard_mat_p_f, expanded_text_embeddings


if __name__ == "__main__":
    torch.manual_seed(0)

    text_channels = 10
    mel_channels = 20
    attention_dim = 128
    attention_head = 8
    dropout = 0.1
    noise_scale = 2.0

    aligner = RoMoAligner(
        text_channels, mel_channels, attention_dim, attention_head, dropout, noise_scale
    )

    batch_size = 2
    text_len = 5
    mel_len = 30

    text_embeddings = torch.randn(batch_size, text_len, text_channels)
    mel_embeddings = torch.randn(batch_size, mel_len, mel_channels)
    text_mask = torch.ones(batch_size, text_len).bool()
    mel_mask = torch.ones(batch_size, mel_len).bool()
    text_mask[1, 3:] = False
    mel_mask[1, 7:] = False

    soft_alignment, hard_alignment, expanded_text_embeddings = aligner(
        text_embeddings,
        mel_embeddings,
        text_mask,
        mel_mask,
        direction=["forward", "backward"],
    )

    print("Soft alignment shape:", soft_alignment.shape)
    print("Hard alignment shape:", hard_alignment.shape)
    print("Expanded text embeddings shape:", expanded_text_embeddings.shape)
