from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from rough_aligner import RoughAligner
from mobo_aligner import MoBoAligner
import robo_utils
from tensor_utils import get_valid_max, get_mat_p_f


def get_indices(int_dur, text_mask, num_context_frames=3):
    batch_size = int_dur.shape[0]

    boundary_index = (int_dur.cumsum(1) - 1) * text_mask
    offsets = (
        torch.arange(-num_context_frames, num_context_frames + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    indices = boundary_index.unsqueeze(1) + offsets

    min_indices, max_indices = get_valid_max(boundary_index, text_mask)
    min_indices = min_indices.unsqueeze(1).unsqueeze(2)
    max_indices = max_indices.unsqueeze(1).unsqueeze(2)

    indices = torch.clamp(indices, min=min_indices, max=max_indices)
    indices = indices.view(batch_size, -1)

    unique_indices = (torch.unique(i) for i in indices)
    unique_indices = torch.nn.utils.rnn.pad_sequence(
        unique_indices, batch_first=True, padding_value=-1
    )

    unique_indices_mask = unique_indices != -1
    unique_indices = unique_indices * unique_indices_mask

    return unique_indices, unique_indices_mask


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
        Selects the corresponding mel_embeddings according to the boundary indices predicted by the rough aligner.

        Args:
            mel_embeddings (torch.Tensor): The original mel feature sequence, with a shape of (B, J, C).
            selected_boundary_indices (torch.Tensor): The indices near the boundaries predicted by the rough aligner, with a shape of (B, K).

        Returns:
            torch.Tensor: The selected mel feature sequence, with a shape of (B, K, C).
        """
        hidden_size = mel_embeddings.shape[2]

        selected_mel_embeddings = torch.gather(
            mel_embeddings,
            1,
            selected_boundary_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
        )

        selected_mel_embeddings = (
            selected_mel_embeddings * selected_boundary_indices_mask.unsqueeze(-1)
        )

        return selected_mel_embeddings

    def get_possible_boundaries(
        self, durations_normalized, text_mask, mel_mask, D=3
    ):
        # Calculate the possible boundaries of each text token based on the results of the rough aligner
        # if the length of text tokens is I, the number of possible boundaries is about I*(2*D+1)
        T = mel_mask.sum(dim=1)
        float_dur = durations_normalized * T.unsqueeze(1)
        int_dur = robo_utils.float_to_int_duration(float_dur, T, text_mask)
        selected_boundary_indices, selected_boundary_indices_mask = get_indices(
            int_dur, text_mask, D
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
    ) -> Tuple[
        Optional[torch.FloatTensor], Optional[torch.FloatTensor], torch.FloatTensor
    ]:
        durations_normalized = self.rough_aligner(
            text_embeddings, mel_embeddings, text_mask, mel_mask
        )

        selected_boundary_indices, selected_boundary_indices_mask = (
            self.get_possible_boundaries(
                durations_normalized, text_mask, mel_mask, D=3
            )
        )

        # Select the corresponding mel_embeddings based on the possible boundary indices
        selected_mel_embeddings = self.select_mel_embeddings(
            mel_embeddings, selected_boundary_indices, selected_boundary_indices_mask
        )

        # Run a fine-grained MoBoAligner
        mat_p_d, hard_alignment, expanded_text_embeddings = self.mobo_aligner(
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

        return mat_p_f, hard_alignment, expanded_text_embeddings


if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(0)

    # 定义输入维度和隐藏层维度
    text_channels = 10
    mel_channels = 20
    attention_dim = 128
    attention_head = 8
    dropout = 0.1
    noise_scale = 2.0

    # 创建RoMoAligner实例
    aligner = RoMoAligner(
        text_channels, mel_channels, attention_dim, attention_head, dropout, noise_scale
    )

    # 生成随机输入数据
    batch_size = 2
    text_len = 5
    mel_len = 30

    text_embeddings = torch.randn(batch_size, text_len, text_channels)
    mel_embeddings = torch.randn(batch_size, mel_len, mel_channels)
    text_mask = torch.ones(batch_size, text_len).bool()
    mel_mask = torch.ones(batch_size, mel_len).bool()
    text_mask[1, 3:] = False
    mel_mask[1, 7:] = False

    # 运行RoMoAligner
    soft_alignment, hard_alignment, expanded_text_embeddings = aligner(
        text_embeddings,
        mel_embeddings,
        text_mask,
        mel_mask,
        direction=["forward", "backward"],
    )

    # 打印结果
    print("Soft alignment shape:", soft_alignment.shape)
    print("Hard alignment shape:", hard_alignment.shape)
    print("Expanded text embeddings shape:", expanded_text_embeddings.shape)
