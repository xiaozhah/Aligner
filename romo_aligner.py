import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from rough_aligner import RoughAligner
from mobo_aligner import MoBoAligner
import robo_utils

def get_indices(int_dur, text_mask, offsets_win_size=3):
    boundry_index = (int_dur.cumsum(1)-1) * text_mask

    batch_size = boundry_index.shape[0]
    offsets = torch.arange(-offsets_win_size, offsets_win_size+1).unsqueeze(0).unsqueeze(-1)
    indices = boundry_index.unsqueeze(1) + offsets

    min_indices = torch.tensor([0] * batch_size).long().unsqueeze(1).unsqueeze(2)
    max_indices = (int_dur.sum(1)-1).unsqueeze(1).unsqueeze(2)
    clamped_indices = torch.clamp(indices, min=min_indices, max=max_indices)
    clamped_indices = clamped_indices.view(batch_size, -1)
    return clamped_indices

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

        mel_embeddings_resampled = self.select_mel_embeddings(
            mel_embeddings, durations_normalized, text_mask, mel_mask
        )

        soft_alignment, hard_alignment, expanded_text_embeddings = self.mobo_aligner(
            text_embeddings,
            mel_embeddings_resampled,
            text_mask,
            mel_mask,
            direction,
            return_hard_alignment=True,
        )

        return soft_alignment, hard_alignment, expanded_text_embeddings

    def select_mel_embeddings(
        self, mel_embeddings, durations_normalized, text_mask, mel_mask
    ):
        """
        Select several boundary indices of mel_embeddings according to the normalized duration predicted by the rough aligner.

        Args:
            mel_embeddings (torch.Tensor): The original mel feature sequence, shape is (B, J, C).
            durations_normalized (torch.Tensor): The normalized duration proportion of each text token, shape is (B, I).
            text_mask (torch.Tensor): The mask of the text sequence, shape is (B, I).
            mel_mask (torch.Tensor): The mask of the original mel feature sequence, shape is (B, J).

        Returns:
            torch.Tensor: The selected mel feature sequence, shape is (B, I, C).
        """
        # Calculate the number of frames corresponding to each text token
        T = mel_mask.sum(dim=1)
        float_dur = durations_normalized * T.unsqueeze(1)
        int_dur = robo_utils.float_to_int_duration(float_dur, T, text_mask)

        selected_boundry_indices = get_indices(int_dur, text_mask)


        return mel_embeddings_selected


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
