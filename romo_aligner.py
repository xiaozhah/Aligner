import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from rough_aligner import RoughAligner
from mobo_aligner import MoBoAligner


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

        mel_embeddings_resampled = self.resample_mel_embeddings(
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

    def resample_mel_embeddings(
        self, mel_embeddings, durations_normalized, text_mask, mel_mask
    ):
        """
        根据粗略对齐器预测的归一化时长,对mel_embeddings进行重采样。

        Args:
            mel_embeddings (torch.Tensor): 原始的mel特征序列,shape为(B, J, C)。
            durations_normalized (torch.Tensor): 归一化的每个文本token的时长占比,shape为(B, I)。
            text_mask (torch.Tensor): 文本序列的mask,shape为(B, I)。
            mel_mask (torch.Tensor): 原始mel特征序列的mask,shape为(B, J)。

        Returns:
            torch.Tensor: 重采样后的mel特征序列,shape为(B, I, C)。
        """
        # 根据durations_normalized和text_mask计算每个文本token对应的帧数
        T = mel_mask.sum(dim=1)
        text_durations = durations_normalized * T.unsqueeze(1)

        text_durations = text_durations.round().long()  # TODO: Fix this in the future

        # 根据mel_lengths对mel_embeddings进行重采样
        mel_embeddings_resampled = []
        for i in range(mel_embeddings.size(0)):
            mel_embedding = mel_embeddings[i, : text_durations[i].max()]
            mel_embedding_resampled = (
                F.interpolate(
                    mel_embedding.transpose(0, 1).unsqueeze(0),
                    size=text_durations[i],
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            mel_embeddings_resampled.append(mel_embedding_resampled)

        mel_embeddings_resampled = torch.nn.utils.rnn.pad_sequence(
            mel_embeddings_resampled, batch_first=True
        )

        return mel_embeddings_resampled


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
