import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

from layers import LinearNorm
import robo_utils


class RoughAligner(nn.Module):
    def __init__(self, attention_dim, attention_head, dropout):
        super(RoughAligner, self).__init__()

        self.cross_attention = MultiHeadedAttention(
            attention_head, attention_dim, dropout
        )
        self.final_layer = LinearNorm(attention_head, 1)

    @torch.no_grad()
    def norm_dur(self, log_dur, text_mask, T):
        """
        make the sum of dur exactly equal to T
        """
        log_dur_masked = log_dur.masked_fill(~text_mask, -float("inf"))
        float_dur = (
            log_dur_masked - log_dur_masked.logsumexp(1, keepdims=True)
        ).exp() * T.unsqueeze(1)
        return float_dur

    def forward(self, text_embeddings, mel_embeddings, text_mask, mel_mask):
        """
        Compute the normalized durations of each text token based on the cross-attention of the text and mel embeddings.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, H).
            mel_embeddings (torch.Tensor): The mel embeddings of shape (B, J, H).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel mask of shape (B, J).

        Returns:
            log_dur (torch.Tensor): The float durations of each text token, with shape (B, I), which the sum of each row may not be equal to the corresponding mel length.
            int_dur (torch.Tensor): The integer durations of each text token, with shape (B, I), which the sum of each row is equal to the corresponding mel length.
        """

        self.cross_attention(
            mel_embeddings, text_embeddings, text_embeddings, ~text_mask.unsqueeze(1)
        )
        attn = self.cross_attention.attn
        x = attn.sum(2).transpose(1, 2) # (B, I, head)
        x = self.final_layer(x).squeeze(-1)
        log_dur = F.relu(x) * text_mask

        T = mel_mask.sum(dim=1)
        float_dur = self.norm_dur(log_dur, text_mask, T)
        int_dur = robo_utils.float_to_int_duration(float_dur, T, text_mask)

        return log_dur, int_dur


if __name__ == "__main__":
    torch.manual_seed(0)

    attention_dim = 128
    attention_head = 8
    dropout = 0.1

    aligner = RoughAligner(attention_dim, attention_head, dropout)

    batch_size = 2
    text_len = 5
    audio_len = 30

    text_embeddings = torch.randn(batch_size, text_len, attention_dim)
    audio_embeddings = torch.randn(batch_size, audio_len, attention_dim)
    text_mask = torch.ones(batch_size, text_len).bool()
    audio_mask = torch.ones(batch_size, audio_len).bool()

    text_mask[1, 3:] = False
    audio_mask[1, 7:] = False

    durations_normalized = aligner(
        text_embeddings, audio_embeddings, text_mask, audio_mask
    )
    print(durations_normalized)
