import os
import numpy as np
from scipy.stats import betabinom

# from https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/helpers.py

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    """
    Calculate the Beta-Binomial prior distribution for alignment.
    
    Args:
        phoneme_count (int): Number of phonemes.
        mel_count (int): Number of mel spectrogram frames.
        scaling_factor (float): Scaling factor for the distribution, default is 1.0.
    
    Returns:
        np.array: 2D array of prior probabilities [mel_count, phoneme_count].
    """
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)

def compute_attn_prior(x_len, y_len, scaling_factor=1.0):
    """
    Compute attention priors for the alignment network.
    
    Args:
        x_len (int): Length of input sequence (e.g., number of phonemes).
        y_len (int): Length of output sequence (e.g., number of mel frames).
        scaling_factor (float): Scaling factor for the distribution.
    
    Returns:
        np.array: Attention prior matrix [y_len, x_len].
    """
    attn_prior = beta_binomial_prior_distribution(
        x_len,
        y_len,
        scaling_factor,
    )
    return attn_prior  # [y_len, x_len]

def load_or_compute_attn_prior(self, token_ids, wav, rel_wav_path):
    """
    Load or compute and save the attention prior.
    
    Args:
        token_ids (list): Input token IDs.
        wav (np.array): Waveform data.
        rel_wav_path (str): Relative path to the wav file.
    
    Returns:
        np.array: Attention prior matrix.
    """
    attn_prior_file = os.path.join(self.attn_prior_cache_path, f"{rel_wav_path}.npy")
    
    if os.path.exists(attn_prior_file):
        # If cached prior exists, load and return it
        return np.load(attn_prior_file)
    else:
        # Compute the prior, save it, and return
        token_len = len(token_ids)
        mel_len = wav.shape[1] // self.ap.hop_length
        attn_prior = compute_attn_prior(token_len, mel_len)
        np.save(attn_prior_file, attn_prior)
        return attn_prior
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test parameters
    phoneme_count = 50
    mel_count = 200
    scaling_factor = 1.0

    # Compute attention prior
    attn_prior = compute_attn_prior(phoneme_count, mel_count, scaling_factor)

    # Visualize the attention prior
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_prior, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title(f'Attention Prior (Phonemes: {phoneme_count}, Mel frames: {mel_count})')
    plt.xlabel('Phoneme index')
    plt.ylabel('Mel frame index')
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Shape of attention prior: {attn_prior.shape}")
    print(f"Min value: {attn_prior.min():.6f}")
    print(f"Max value: {attn_prior.max():.6f}")
    print(f"Mean value: {attn_prior.mean():.6f}")

    # Test with different scaling factor
    scaling_factor = 0.5
    attn_prior_scaled = compute_attn_prior(phoneme_count, mel_count, scaling_factor)

    # Visualize the scaled attention prior
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_prior_scaled, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title(f'Scaled Attention Prior (SF: {scaling_factor})')
    plt.xlabel('Phoneme index')
    plt.ylabel('Mel frame index')
    plt.tight_layout()
    plt.show()

    # Compare statistics
    print(f"\nWith scaling factor {scaling_factor}:")
    print(f"Min value: {attn_prior_scaled.min():.6f}")
    print(f"Max value: {attn_prior_scaled.max():.6f}")
    print(f"Mean value: {attn_prior_scaled.mean():.6f}")