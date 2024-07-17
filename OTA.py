import torch
from torch import nn

from aligner import AlignmentNetwork
from typing import Tuple
from monotonic_align import maximum_path
from loss import ForwardSumLoss

class OTAligner(nn.Module):
    def __init__(self, mel_channels, text_channels, blank_logprob=-1):
        super(OTAligner, self).__init__()

        self.aligner = AlignmentNetwork(
            in_query_channels=mel_channels,
            in_key_channels=text_channels,
        )
        self.aligner_loss = ForwardSumLoss(blank_logprob=blank_logprob)
        self.blank_logprob = blank_logprob

    def interleave_with_blank(self, aligner_soft):
        bsz, J, _ = aligner_soft.shape
        x = (
            torch.stack(
                (aligner_soft, torch.full_like(aligner_soft, self.blank_logprob)), dim=2
            )
            .transpose(-2, -1)
            .reshape(bsz, J, -1)
        ) # interleave with blank
        x = torch.cat((torch.full((2, 30, 1), fill_value=-1), x), dim=-1) # pad blank at the beginning
        return x

    def _forward_aligner(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        x_mask: torch.IntTensor,
        y_mask: torch.IntTensor,
        attn_priors: torch.FloatTensor,
    ) -> Tuple[
        torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Aligner forward pass.
        from https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/models/forward_tts.py#L525

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.
            attn_priors (torch.FloatTensor): Prior for the aligner network map.

        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, I, C_text]`
            - y: :math:`[B, J, C_mel]`
            - x_mask: :math:`[B, 1, I]`
            - y_mask: :math:`[B, 1, J]`
            - attn_priors: :math:`[B, J, I]`

            - aligner_durations: :math:`[B, I]`
            - aligner_soft: :math:`[B, J, I]`
            - aligner_logprob: :math:`[B, J, I]`
            - aligner_mas: :math:`[B, J, I]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(
            y_mask, 2
        )  # [B, 1, I, J]
        aligner_soft, aligner_logprob = self.aligner(
            y.transpose(1, 2), x.transpose(1, 2), x_mask, attn_priors
        )
        aligner_mas = maximum_path(
            aligner_soft.transpose(1, 2).contiguous(),
            attn_mask.squeeze(1).contiguous(),
        )
        aligner_durations = torch.sum(aligner_mas, -1).int()
        aligner_mas = aligner_mas.transpose(
            1, 2
        )  # [B, T_max, T_max2] -> [B, T_max2, T_max]
        return aligner_durations, aligner_soft, aligner_logprob, aligner_mas

    def forward(self, text_embeddings, mel_embeddings, text_mask, mel_mask, attn_priors):
        """
        Compute the alignments.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, D_text).
            mel_embeddings (torch.Tensor): The mel embeddings of shape (B, J, D_mel).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel mask of shape (B, J).
        """
        # Alignment network and durations
        aligner_durations, aligner_soft, aligner_logprob, aligner_mas = (
            self._forward_aligner(
                x=text_embeddings,
                y=mel_embeddings,
                x_mask=text_mask.unsqueeze(1),
                y_mask=mel_mask.unsqueeze(1),
                attn_priors=attn_priors,
            )
        )

        aligner_loss = self.aligner_loss(aligner_logprob, text_mask.sum(1), mel_mask.sum(1))

        return {
            "aligner_durations": aligner_durations,
            "aligner_soft": aligner_soft,
            "aligner_logprob": aligner_logprob,
            "aligner_mas": aligner_mas,
            "aligner_loss": aligner_loss
        }

if __name__ == "__main__":
    batch_size = 2
    text_len = 5
    audio_len = 30
    mel_dim = 80
    text_dim = 128

    text_embeddings = torch.randn(batch_size, text_len, text_dim)
    audio_embeddings = torch.randn(batch_size, audio_len, mel_dim)
    text_mask = torch.ones(batch_size, text_len).bool()
    audio_mask = torch.ones(batch_size, audio_len).bool()

    text_mask[1, 3:] = False
    audio_mask[1, 7:] = False

    aligner = OTAligner(mel_channels=mel_dim, text_channels=text_dim)

    durations_normalized = aligner(
        text_embeddings, audio_embeddings, text_mask, audio_mask, None
    )
    print(durations_normalized)