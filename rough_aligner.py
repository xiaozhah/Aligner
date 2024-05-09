import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

from layers import LinearNorm
import robo_utils


class RoughAligner(nn.Module):
    def __init__(
        self, text_channels, mel_channels, attention_dim, attention_head, dropout
    ):
        super(RoughAligner, self).__init__()

        self.text_layer = LinearNorm(text_channels, attention_dim)
        self.mel_layer = LinearNorm(mel_channels, attention_dim)
        self.cross_attention = MultiHeadedAttention(
            attention_head, attention_dim, dropout
        )
        self.final_layer = LinearNorm(attention_dim, 1)

    def forward(self, text_embeddings, mel_embeddings, text_mask, mel_mask):
        """
        Compute the normalized durations of each text token based on the cross-attention of the text and mel embeddings.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, C1).
            mel_embeddings (torch.Tensor): The mel embeddings of shape (B, J, C2).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel mask of shape (B, J).

        Returns:
            torch.Tensor: The normalized durations of each text token of shape (B, I).
        """

        text_hidden = self.text_layer(text_embeddings)  # (B, I, H)
        mel_hidden = self.mel_layer(mel_embeddings)  # (B, J, H)

        x = self.cross_attention(text_hidden, mel_hidden, mel_hidden, ~mel_mask.unsqueeze(1))
        x = x * text_mask.unsqueeze(-1)
        x = self.final_layer(x).squeeze(-1)
        x = F.sigmoid(x) * text_mask
        norm_dur = x / x.sum(dim=1, keepdim=True)

        T = mel_mask.sum(dim=1)
        float_dur = norm_dur * T.unsqueeze(1)
        int_dur = robo_utils.float_to_int_duration(float_dur, T, text_mask)

        return float_dur, int_dur


if __name__ == "__main__":
    torch.manual_seed(0)

    text_channels = 10
    audio_channels = 20
    attention_dim = 128
    attention_head = 8
    dropout = 0.1

    aligner = RoughAligner(
        text_channels, audio_channels, attention_dim, attention_head, dropout
    )

    batch_size = 2
    text_len = 5
    audio_len = 30

    text_embeddings = torch.randn(batch_size, text_len, text_channels)
    audio_embeddings = torch.randn(batch_size, audio_len, audio_channels)
    text_mask = torch.ones(batch_size, text_len).bool()
    audio_mask = torch.ones(batch_size, audio_len).bool()

    text_mask[1, 3:] = False
    audio_mask[1, 7:] = False

    durations_normalized = aligner(
        text_embeddings, audio_embeddings, text_mask, audio_mask
    )
    print(durations_normalized)
