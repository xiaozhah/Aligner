import torch
import torch.nn as nn
import math
from roll import roll_tensor


class MoBoAligner(nn.Module):
    def __init__(self, temperature_min=0.1, temperature_max=1.0):
        super(MoBoAligner, self).__init__()
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

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

    def apply_gumbel_noise(self, energy, temperature_ratio):
        """
        Apply Gumbel noise and temperature to the energy matrix.

        Args:
            energy (torch.Tensor): The energy matrix of shape (B, I, J).
            temperature_ratio (float): The temperature ratio for Gumbel noise.

        Returns:
            torch.Tensor: The energy matrix with Gumbel noise and temperature applied.
        """
        temperature = (
            self.temperature_min
            + (self.temperature_max - self.temperature_min) * temperature_ratio
        )
        noise = -torch.log(-torch.log(torch.rand_like(energy)))
        return (energy + noise) / temperature

    def compute_log_cond_prob(self, energy, text_mask, mel_mask, direction="alpha"):
        """
        Compute the log conditional probability of the alignment in the specified direction.

        Args:
            energy (torch.Tensor): The energy matrix of shape (B, I, J).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).
            direction (str): The direction of the alignment, either "alpha" or "beta".

        Returns:
            torch.Tensor: The log conditional probability tensor of shape (B, I, J, K),
                where K is J for "alpha" and J-1 for "beta".
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        if direction == "alpha":
            energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, J)  # (B, I, J, K)
            triu = torch.triu(torch.ones((J, J)), diagonal=0).to(
                energy.device
            )  # (K, J)
        elif direction == "beta":
            energy_4D = energy.unsqueeze(-1).repeat(1, 1, 1, J - 1)  # (B, I, J, K)
            triu = torch.tril(torch.ones((J - 1, J)), diagonal=0).to(
                energy.device
            )  # (K, J), K == J-1
        else:
            raise ValueError("direction must be 'alpha' or 'beta'")

        triu = triu.unsqueeze(-1).unsqueeze(0)  # (1, K, J, 1)
        triu = triu.repeat(B, 1, 1, I)  # (B, K, J, I)
        triu = triu.transpose(1, 3)  # (B, I, J, K)

        mask = (
            text_mask.unsqueeze(2).unsqueeze(3)
            * mel_mask.unsqueeze(1).unsqueeze(3)
            * mel_mask.unsqueeze(1).unsqueeze(1)
        )
        if direction == "beta":  # because K is J-1
            mask = mask[:, :, :, :-1]
        triu = triu * mask

        mask_invalid = (
            text_mask.unsqueeze(2).unsqueeze(3) * mel_mask.unsqueeze(1).unsqueeze(1)
        ).repeat(1, 1, J, 1)
        text_invalid = text_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, J, J)
        if direction == "beta":  # because K is J-1
            mask_invalid = mask_invalid[:, :, :, :-1]
            text_invalid = text_invalid[:, :, :, :-1]
        right_mask = mask | (~mask_invalid)

        energy_4D.masked_fill_((triu == 0) & (mask_invalid == 1), -float("Inf"))
        # Fill the energy_4D tensor with -10, to avoid -inf values in the subsequent logsumexp operation.
        energy_4D.masked_fill_(mask_invalid == 0, -10)
        energy_4D = energy_4D - torch.logsumexp(energy_4D, dim=2, keepdim=True)
        energy_4D.masked_fill_(mask_invalid == 0, -float("Inf"))
        energy_4D.masked_fill_(right_mask == 0, -10)
        energy_4D.masked_fill_(text_invalid == 0, -10)

        return energy_4D

    def right_shift(self, x, shifts_text_dim, shifts_mel_dim):
        """
        Shift the tensor x to the right along the text and mel dimensions.

        Args:
            x (torch.Tensor): The input tensor of shape (B, I, J, K).
            shifts_text_dim (torch.Tensor): The shift amounts along the text dimension of shape (B,).
            shifts_mel_dim (torch.Tensor): The shift amounts along the mel dimension of shape (B,).

        Returns:
            torch.Tensor: The right-shifted tensor of shape (B, I, J, K).
        """
        x = roll_tensor(x, shifts=shifts_text_dim, dim=1)
        x = roll_tensor(x, shifts=shifts_mel_dim, dim=2)
        shifts_mel_dim[1:] = shifts_mel_dim[1:] - 1
        x = roll_tensor(x, shifts=shifts_mel_dim, dim=3)
        return x

    def left_shift(self, x, shifts_text_dim, shifts_mel_dim):
        """
        Shift the tensor x to the left along the text and mel dimensions.

        Args:
            x (torch.Tensor): The input tensor of shape (B, I, J).
            shifts_text_dim (torch.Tensor): The shift amounts along the text dimension of shape (B,).
            shifts_mel_dim (torch.Tensor): The shift amounts along the mel dimension of shape (B,).

        Returns:
            torch.Tensor: The left-shifted tensor of shape (B, I, J).
        """
        x = x.unsqueeze(-1)
        x = roll_tensor(x, shifts=-shifts_text_dim, dim=1)
        x = roll_tensor(x, shifts=-shifts_mel_dim, dim=2)
        x = x.squeeze(-1)
        return x

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

    def compute_alpha(self, log_cond_prob_alpha, text_mask, mel_mask):
        """
        Compute alpha recursively in the log domain.

        Args:
            log_cond_prob_alpha (torch.Tensor): The log conditional probability tensor for alpha of shape (B, I, J, K).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The alpha tensor of shape (B, I+1, J+1).
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        alpha = torch.full(
            (B, I + 1, J + 1), -float("inf"), device=log_cond_prob_alpha.device
        )
        alpha[:, 0, 0] = 0  # Initialize alpha[0, 0] = 0
        for i in range(1, I + 1):
            alpha[:, i, i:] = torch.logsumexp(
                alpha[:, i - 1, :-1].unsqueeze(1)
                + log_cond_prob_alpha[:, i - 1, (i - 1) :],
                dim=2,
            )

        return alpha

    def compute_beta(self, log_cond_prob_beta, text_mask, mel_mask):
        """
        Compute beta recursively in the log domain.

        Args:
            log_cond_prob_beta (torch.Tensor): The log conditional probability tensor for beta of shape (B, I, J, K).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The beta tensor of shape (B, I, J).
        """
        B, I = text_mask.shape
        _, J = mel_mask.shape

        beta = torch.full((B, I, J), -float("inf"), device=log_cond_prob_beta.device)
        beta[:, -1, -1] = 0  # Initialize beta_{I,J} = 1
        for i in range(I - 2, -1, -1):
            beta[:, i, : (J + i - I + 1)] = torch.logsumexp(
                beta[:, i + 1, 1:].unsqueeze(1)
                + log_cond_prob_beta[:, i, : (J + i - I + 1)],
                dim=2,
            )

        return beta

    def compute_gamma(self, alpha, beta, text_mask, mel_mask):
        """
        Compute gamma (soft alignment) in the log domain.

        Args:
            alpha (torch.Tensor): The alpha tensor of shape (B, I+1, J+1).
            beta (torch.Tensor): The beta tensor of shape (B, I, J).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The gamma tensor of shape (B, I, J) in the log domain.
        """
        _, J = mel_mask.shape

        gamma = alpha[:, 1:, 1:] + beta

        gamma_mask = text_mask.unsqueeze(2) * mel_mask.unsqueeze(1)
        gamma_logsumexp_mask = text_mask.unsqueeze(2).repeat(1, 1, J) | gamma_mask
        gamma.masked_fill_(gamma_logsumexp_mask == 0, -float("Inf"))
        gamma = gamma - torch.logsumexp(gamma, dim=1, keepdim=True)
        gamma.masked_fill_(gamma_mask == 0, -float("Inf"))

        return gamma, gamma_mask

    def compute_expanded_text_embeddings(self, gamma, text_embeddings, mel_mask):
        """
        Compute the expanded text embeddings based on gamma and text embeddings.

        Args:
            gamma (torch.Tensor): The gamma tensor of shape (B, I, J) in the log domain.
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, D_text).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).

        Returns:
            torch.Tensor: The expanded text embeddings of shape (B, J, D_text).
        """
        expanded_text_embeddings = torch.bmm(
            torch.exp(gamma).transpose(1, 2), text_embeddings
        )
        expanded_text_embeddings = expanded_text_embeddings * mel_mask.unsqueeze(2)

        return expanded_text_embeddings

    def forward(
        self, text_embeddings, mel_embeddings, text_mask, mel_mask, temperature_ratio
    ):
        """
        Compute the soft alignment (gamma) and the expanded text embeddings.

        Args:
            text_embeddings (torch.Tensor): The text embeddings of shape (B, I, D_text).
            mel_embeddings (torch.Tensor): The mel spectrogram embeddings of shape (B, J, D_mel).
            text_mask (torch.Tensor): The text mask of shape (B, I).
            mel_mask (torch.Tensor): The mel spectrogram mask of shape (B, J).
            temperature_ratio (float): The temperature ratio for Gumbel noise.

        Returns:
            tuple: A tuple containing:
                - gamma (torch.Tensor): The soft alignment tensor of shape (B, I, J) in the log domain.
                - expanded_text_embeddings (torch.Tensor): The expanded text embeddings of shape (B, J, D_text).
        """
        # Compute the energy matrix
        energy = self.compute_energy(text_embeddings, mel_embeddings)

        # Apply Gumbel noise and temperature
        energy = self.apply_gumbel_noise(energy, temperature_ratio)

        # Compute the log conditional probability P(B_i=j | B_{i-1}=k) for alpha
        log_cond_prob_alpha = self.compute_log_cond_prob(
            energy, text_mask, mel_mask, direction="alpha"
        )  # (B, I, J, K)

        # Compute the log conditional probability P(B_i=j | B_{i+1}=k) for beta
        log_cond_prob_beta = self.compute_log_cond_prob(
            energy, text_mask, mel_mask, direction="beta"
        )  # (B, I, J, K)
        log_cond_prob_beta = self.right_shift(
            log_cond_prob_beta,
            shifts_text_dim=self.compute_max_length_diff(text_mask),
            shifts_mel_dim=self.compute_max_length_diff(mel_mask),
        )

        # Compute alpha recursively in the log domain
        alpha = self.compute_alpha(log_cond_prob_alpha, text_mask, mel_mask)

        # Compute beta recursively in the log domain
        beta = self.compute_beta(log_cond_prob_beta, text_mask, mel_mask)
        beta = self.left_shift(
            beta,
            shifts_text_dim=self.compute_max_length_diff(text_mask),
            shifts_mel_dim=self.compute_max_length_diff(mel_mask),
        )

        # Compute gamma (soft alignment) in the log domain
        gamma, gamma_mask = self.compute_gamma(alpha, beta, text_mask, mel_mask)

        # Compute the expanded text embeddings
        expanded_text_embeddings = self.compute_expanded_text_embeddings(
            gamma, text_embeddings, mel_mask
        )

        return gamma, gamma_mask, expanded_text_embeddings  # gamma is still in the log domain
