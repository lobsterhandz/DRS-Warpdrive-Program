import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Define the core model
def stress_energy_tensor(x, t, lambda_, dim_factor, perturbation=0):
    base = np.cos(lambda_ * x) + np.sin(t)
    perturb = perturbation * np.sin(2 * np.pi * t)  # Apply perturbation
    dimensional_term = dim_factor * np.cos(np.pi * x / dim_factor)
    return base + perturb + dimensional_term

# Frequency analysis
def frequency_analysis(x, t, lambda_, dim_factor):
    tensor = stress_energy_tensor(x, t, lambda_, dim_factor)
    fft_result = np.fft.fft(tensor)
    freq = np.fft.fftfreq(len(tensor), d=(x[1] - x[0]))
    return freq, np.abs(fft_result)

# Compute entropy and variance
def compute_entropy_variance(tensor):
    prob = np.abs(tensor) / np.sum(np.abs(tensor))
    entropy = -np.sum(prob * np.log(prob + 1e-12))  # Add small epsilon to avoid log(0)
    variance = np.var(tensor)
    return entropy, variance

# Compare to known systems (dummy example)
def compare_to_known_systems(tensor):
    reference = np.cos(np.linspace(0, 10, len(tensor)))  # Example reference system
    comparison = np.corrcoef(tensor, reference)[0, 1]
    return comparison

# Unified analysis function
def analyze_system(x, t, lambdas, dim_factors, perturbation=0):
    results = {"entropy": [], "variance": [], "frequency": [], "comparison": []}
    
    for lambda_ in lambdas:
        for dim_factor in dim_factors:
            tensor = stress_energy_tensor(x, t, lambda_, dim_factor, perturbation)
            entropy, variance = compute_entropy_variance(tensor)
            comparison = compare_to_known_systems(tensor)
            freq, fft_result = frequency_analysis(x, t, lambda_, dim_factor)
            
            # Save results
            results["entropy"].append((lambda_, dim_factor, entropy))
            results["variance"].append((lambda_, dim_factor, variance))
            results["frequency"].append((lambda_, dim_factor, freq, fft_result))
            results["comparison"].append((lambda_, dim_factor, comparison))
    
    return results

# Visualization function
def visualize_results(results):
    # Entropy vs Lambda
    lambdas = [e[0] for e in results["entropy"]]
    entropies = [e[2] for e in results["entropy"]]
    plt.scatter(lambdas, entropies, c='yellow', label="Entropy")
    plt.colorbar(label="Dim Factor")
    plt.xlabel("Nonlinear Coupling Strength (lambda)")
    plt.ylabel("Entropy")
    plt.title("Entropy vs Lambda")
    plt.show()
    
    # Variance vs Lambda
    variances = [v[2] for v in results["variance"]]
    plt.scatter(lambdas, variances, c='orange', label="Variance")
    plt.colorbar(label="Dim Factor")
    plt.xlabel("Nonlinear Coupling Strength (lambda)")
    plt.ylabel("Variance")
    plt.title("Variance vs Lambda")
    plt.show()
    
    # Frequency Spectrum
    for freq_data in results["frequency"]:
        lambda_, dim_factor, freq, fft_result = freq_data
        plt.plot(freq, fft_result, label=f"Lambda={lambda_}, Dim={dim_factor}")
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum")
    plt.show()
    
    # Comparison to known systems
    comparisons = [c[2] for c in results["comparison"]]
    plt.bar(range(len(comparisons)), comparisons, color='blue')
    plt.xlabel("Case Index")
    plt.ylabel("Comparison Correlation")
    plt.title("Comparison to Known Systems")
    plt.show()

# Parameters
x = np.linspace(0, 10, 100)
t = np.linspace(0, 10, 100)
lambdas = np.linspace(0.1, 1.0, 5)
dim_factors = np.linspace(0.1, 0.5, 5)

# Run analysis
results = analyze_system(x, t, lambdas, dim_factors, perturbation=0.1)
visualize_results(results)
