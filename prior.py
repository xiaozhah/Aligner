import numpy as np
from scipy.stats import betabinom
import time

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

# Refined NumPy implementation
def refined_beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    m = np.arange(1, M + 1)[:, np.newaxis]
    x = np.arange(P)
    a = scaling_factor * m
    b = scaling_factor * (M + 1 - m)
    probs = betabinom.pmf(x, P, a, b)
    return probs

# Cython implementation (assuming it's compiled and imported)
# from beta_binomial_cython import cython_beta_binomial_prior_distribution

# Unified interface
def compute_attn_prior(phoneme_count, mel_count, scaling_factor=1.0, method='refined'):
    methods = {
        'original': original_beta_binomial_prior_distribution,
        'refined': refined_beta_binomial_prior_distribution,
        # 'cython': cython_beta_binomial_prior_distribution  # Uncomment if Cython version is available
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available methods are: {', '.join(methods.keys())}")
    
    return methods[method](phoneme_count, mel_count, scaling_factor)

# Benchmark function
def benchmark(method, phoneme_count, mel_count, scaling_factor=1.0, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        result = compute_attn_prior(phoneme_count, mel_count, scaling_factor, method=method)
        end = time.time()
        times.append(end - start)
    return np.mean(times), result

# Function to compare outputs
def compare_outputs(original_output, test_output, tolerance=1e-10):
    return np.allclose(original_output, test_output, rtol=tolerance, atol=tolerance)

# Main benchmark script
def run_benchmark():
    test_cases = [
        (10, 20),    # Small input
        (50, 100),   # Medium input
        (200, 400),  # Large input
        (500, 1000)  # Very large input
    ]

    methods = ['original', 'refined']  # Add 'cython' if available
    
    print("Phonemes | Mels | Method  | Time (s) | Speedup | Outputs Match")
    print("---------|------|---------|----------|---------|---------------")

    for P, M in test_cases:
        original_time, original_output = benchmark('original', P, M)
        
        for method in methods:
            time_taken, output = benchmark(method, P, M)
            speedup = original_time / time_taken if method != 'original' else 1.0
            outputs_match = compare_outputs(original_output, output)
            
            print(f"{P:8d} | {M:4d} | {method:7s} | {time_taken:.6f} | {speedup:.2f}x | {outputs_match}")
        
        print("---------|------|---------|----------|---------|---------------")

    print("\nVerifying output consistency across all methods...")
    all_match = all(
        compare_outputs(
            compute_attn_prior(P, M, method='original'),
            compute_attn_prior(P, M, method=method)
        )
        for P, M in test_cases
        for method in methods
    )
    
    if all_match:
        print("All outputs match within the specified tolerance.")
    else:
        print("WARNING: Not all outputs match. Please review the implementations for potential discrepancies.")

def plot():
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

if __name__ == "__main__":
    run_benchmark()
    plot()