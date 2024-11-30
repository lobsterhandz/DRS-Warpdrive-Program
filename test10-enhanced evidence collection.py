import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Constants
gamma = 0.2  # Damping factor
omega = 2.0  # Oscillatory frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
rho_n = 1.0  # Energy density
time = np.linspace(0, 10, 1000)
x = np.linspace(0, 10, 1000)

# Extended parameter ranges
lambda_res_values = np.linspace(0.5, 1.0, 6)  # Nonlinear coupling strengths
higher_dim_factors = np.linspace(0.1, 0.4, 6)  # Higher-dimensional scaling factors
n_max = 20

# Function to compute divergence
def compute_divergence(lambda_res, higher_dim_factor):
    T_time = np.zeros((len(time), len(x)))
    T_res = np.zeros((len(time), len(x)))
    T_extra = np.zeros((len(time), len(x)))

    # Damped oscillations
    for n in range(1, n_max + 1):
        T_time += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + x[None, :])

    # Nonlinear product term
    T_res += lambda_res * np.exp(-gamma * time[:, None]) * np.cos(phi * omega * time[:, None] + x[None, :])

    # Higher-dimensional term
    T_extra += higher_dim_factor * np.exp(-gamma * time[:, None]) * np.cos(omega * x[None, :] + time[:, None])

    # Total divergence
    T_total = T_time + T_res + T_extra
    divergence = np.gradient(T_total, time, axis=0) + np.gradient(T_total, x, axis=1)
    return divergence

# Analyze parameter space
results = []
for lambda_res in lambda_res_values:
    for higher_dim_factor in higher_dim_factors:
        divergence = compute_divergence(lambda_res, higher_dim_factor)
        mean_divergence = np.mean(np.abs(divergence))
        results.append((lambda_res, higher_dim_factor, mean_divergence))

        # Plot divergence map
        plt.figure(figsize=(10, 6))
        plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', cmap='coolwarm')
        plt.colorbar(label="Divergence")
        plt.title(f"Divergence Map ($\\lambda_{{res}}$={lambda_res}, Higher-Dim Factor={higher_dim_factor})")
        plt.xlabel("Space (x)")
        plt.ylabel("Time (t)")
        plt.show()

# Aggregate results
results = np.array(results)

# Sensitivity plot
plt.figure(figsize=(10, 6))
for higher_dim_factor in higher_dim_factors:
    mask = results[:, 1] == higher_dim_factor
    plt.plot(results[mask, 0], results[mask, 2], marker='o', label=f"Higher-Dim Factor = {higher_dim_factor:.2f}")

plt.title("Mean Absolute Divergence Across Parameters")
plt.xlabel("Nonlinear Coupling Strength ($\\lambda_{res}$)")
plt.ylabel("Mean Absolute Divergence")
plt.legend()
plt.grid()
plt.show()

# Frequency analysis for a specific case
lambda_res = 0.75
higher_dim_factor = 0.2
divergence = compute_divergence(lambda_res, higher_dim_factor)
fft_result = fft(np.mean(divergence, axis=1))

# Plot frequency spectrum
frequencies = np.fft.fftfreq(len(time), d=(time[1] - time[0]))
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(frequencies)//2]))
plt.title(f"Frequency Spectrum ($\\lambda_{{res}}$={lambda_res}, Higher-Dim Factor={higher_dim_factor})")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
