import numpy as np
from scipy.stats import betabinom
import time
import torch

# Original implementation
def original_beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)

# Optimized NumPy implementation
def refined_beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    m = np.arange(1, M + 1)[:, np.newaxis]
    x = np.arange(P)
    a = scaling_factor * m
    b = scaling_factor * (M + 1 - m)
    probs = betabinom.pmf(x, P, a, b)
    return probs

# PyTorch implementation using log-space calculations with batch support
@torch.no_grad()
def pytorch_beta_binomial_prior_distribution(phoneme_counts, mel_counts, scaling_factor=1.0):
    device = phoneme_counts.device
    max_phoneme_count = phoneme_counts.max().item()
    max_mel_count = mel_counts.max().item()

    n = phoneme_counts.unsqueeze(1).unsqueeze(2)
    k = torch.arange(max_phoneme_count, device=device).unsqueeze(0).unsqueeze(0)
    m = torch.arange(1, max_mel_count + 1, device=device).unsqueeze(0).unsqueeze(2)
    
    a = scaling_factor * m
    b = scaling_factor * (mel_counts.unsqueeze(1).unsqueeze(2) + 1 - m)
    
    log_coef = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    numerator = torch.lgamma(a + k) + torch.lgamma(b + n - k) + torch.lgamma(a + b)
    denominator = torch.lgamma(a + b + n) + torch.lgamma(a) + torch.lgamma(b)
    
    log_pmf = log_coef + numerator - denominator
    pmf = torch.exp(log_pmf)
    
    # Create a mask to handle different sequence lengths
    mask = (k < n) & (m <= mel_counts.unsqueeze(1).unsqueeze(2))
    pmf.masked_fill_(~mask, 0)
    assert((pmf.min() >= 0).all() & (pmf.max() <= 1).all())
    return pmf

# Unified interface
def compute_attn_prior(phoneme_count, mel_count, scaling_factor=1.0, method='refined', device='cpu'):
    methods = {
        'original': original_beta_binomial_prior_distribution,
        'refined': refined_beta_binomial_prior_distribution,
        'pytorch': lambda p, m, s: pytorch_beta_binomial_prior_distribution(
            torch.tensor([p], dtype=torch.long, device=device),
            torch.tensor([m], dtype=torch.long, device=device),
            s
        )[0].cpu().numpy()
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available methods are: {', '.join(methods.keys())}")
    
    return methods[method](phoneme_count, mel_count, scaling_factor)

# Benchmark function
def benchmark(method, phoneme_count, mel_count, scaling_factor=1.0, runs=5, device='cpu'):
    times = []
    for _ in range(runs):
        start = time.time()
        result = compute_attn_prior(phoneme_count, mel_count, scaling_factor, method=method, device=device)
        end = time.time()
        times.append(end - start)
    return np.mean(times), result

# Function to compare outputs with detailed error analysis
def compare_outputs(original_output, test_output, tolerance=1e-3):
    is_close = np.allclose(original_output, test_output, rtol=tolerance, atol=tolerance)
    max_diff = np.max(np.abs(original_output - test_output))
    mean_diff = np.mean(np.abs(original_output - test_output))
    return is_close, max_diff, mean_diff

# Function to compare all methods
def compare_methods(phoneme_count, mel_count, scaling_factor=1.0, device='cpu'):
    methods = ['original', 'refined', 'pytorch']
    results = {method: compute_attn_prior(phoneme_count, mel_count, scaling_factor, method, device) for method in methods}
    
    print(f"\nComparison for P={phoneme_count}, M={mel_count}:")
    for method1 in methods:
        for method2 in methods[methods.index(method1)+1:]:
            is_close, max_diff, mean_diff = compare_outputs(results[method1], results[method2])
            print(f"{method1} vs {method2}:")
            print(f"  Outputs match: {is_close}")
            print(f"  Max difference: {max_diff:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")

# Main benchmark script
def run_benchmark(device='cpu'):
    test_cases = [
        (10, 20),    # Small input
        (50, 100),   # Medium input
        (200, 400),  # Large input
        (500, 1000)  # Very large input
    ]

    methods = ['original', 'refined', 'pytorch']
    
    print("Phonemes | Mels | Method    | Time (s)  | Speedup | Outputs Match")
    print("---------|------|-----------|-----------|---------|---------------")

    for P, M in test_cases:
        original_time, original_output = benchmark('original', P, M, device=device)
        
        for method in methods:
            time_taken, output = benchmark(method, P, M, device=device)
            speedup = original_time / time_taken if method != 'original' else 1.0
            is_close, _, _ = compare_outputs(original_output, output)
            
            print(f"{P:8d} | {M:4d} | {method:9s} | {time_taken:.6f} | {speedup:.2f}x | {is_close}")
        
        print("---------|------|-----------|-----------|---------|---------------")
        
        compare_methods(P, M, device=device)

# Enhanced visualization function
def plot_distribution(phoneme_count, mel_count, scaling_factor=1.0, device='cpu'):
    import matplotlib.pyplot as plt
    methods = ['original', 'pytorch']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, method in enumerate(methods):
        attn_prior = compute_attn_prior(phoneme_count, mel_count, scaling_factor, method=method, device=device)
        im = axes[i].imshow(attn_prior, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(f'{method.capitalize()} Method')
        axes[i].set_xlabel('Phoneme index')
        axes[i].set_ylabel('Mel frame index')
        fig.colorbar(im, ax=axes[i])
    
    # Plot differences
    diff = compute_attn_prior(phoneme_count, mel_count, scaling_factor, 'pytorch', device) - \
           compute_attn_prior(phoneme_count, mel_count, scaling_factor, 'original', device)
    im = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='coolwarm', norm=plt.Normalize(vmin=-1e-3, vmax=1e-3))
    axes[2].set_title('Difference (PyTorch - Original)')
    axes[2].set_xlabel('Phoneme index')
    axes[2].set_ylabel('Mel frame index')
    fig.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for PyTorch method:")
    attn_prior = compute_attn_prior(phoneme_count, mel_count, scaling_factor, 'pytorch', device)
    print(f"Shape: {attn_prior.shape}")
    print(f"Min value: {attn_prior.min():.6f}")
    print(f"Max value: {attn_prior.max():.6f}")
    print(f"Mean value: {attn_prior.mean():.6f}")

    # Print max difference
    print(f"\nMax absolute difference between PyTorch and Original: {np.abs(diff).max():.6e}")

def batch_example(device='cpu'):
    phoneme_counts = [10, 50, 200, 500]
    mel_counts = [20, 100, 400, 1000]
    scaling_factor = 1.0

    # Convert to PyTorch tensors
    phoneme_counts_tensor = torch.tensor(phoneme_counts, dtype=torch.long, device=device)
    mel_counts_tensor = torch.tensor(mel_counts, dtype=torch.long, device=device)

    batch_result = pytorch_beta_binomial_prior_distribution(phoneme_counts_tensor, mel_counts_tensor, scaling_factor)

    print(f"Batch result shape: {batch_result.shape}")
    
    for i, (p, m) in enumerate(zip(phoneme_counts, mel_counts)):
        sample = batch_result[i]
        print(f"\nSample {i+1} (Phonemes: {p}, Mels: {m}):")
        print(f"  Shape: {sample.shape}")
        print(f"  Min: {sample.min().item():.6f}")
        print(f"  Max: {sample.max().item():.6f}")
        print(f"  Mean: {sample.mean().item():.6f}")

    # Optional: Move result back to CPU for further processing if needed
    batch_result_cpu = batch_result.cpu()
    
    return batch_result_cpu

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_benchmark(device)
    plot_distribution(50, 100, scaling_factor=1.0, device=device)
    batch_example(device)