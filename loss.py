import torch
from torch import nn


class ForwardSumLoss(nn.Module):
    """
    from https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/layers/losses.py#L279C1-L280C1
    """

    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = torch.nn.functional.pad(
            input=attn_logprob.unsqueeze(1), pad=(1, 0), value=self.blank_logprob
        )

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : query_lens[bid], :, : key_lens[bid] + 1
            ]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


def binary_alignment_loss(alignment_hard, alignment_soft):
    """Binary loss that forces soft alignments to match the hard alignments as
    explained in `https://arxiv.org/pdf/2108.10447.pdf`.
    """
    log_sum = torch.log(
        torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)
    ).sum()
    return -log_sum / alignment_hard.sum()
