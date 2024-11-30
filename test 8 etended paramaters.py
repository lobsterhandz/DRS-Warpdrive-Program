import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.2  # Optimal damping factor
omega = 2.0  # Base frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
k = 1.0  # Spatial wave number
rho_n = 1.0  # Base energy density for n = 1
time = np.linspace(0, 10, 500)  # Time array
x = np.linspace(0, 10, 500)  # Space array

# Parameter ranges
lambda_res_values = np.linspace(0.5, 0.8, 4)  # Nonlinear coupling strengths
higher_dim_factors = np.linspace(0.1, 0.25, 4)  # Higher-dimensional scaling factors
n_max = 20  # Number of terms in the summation

# Storage for results
results = []

# Loop through parameters
for higher_dim_factor in higher_dim_factors:
    for lambda_res in lambda_res_values:
        # Initialize components
        T_time = np.zeros((len(time), len(x)))
        T_res = np.zeros((len(time), len(x)))
        T_extra = np.zeros((len(time), len(x)))

        # Compute the damped oscillatory term
        for n in range(1, n_max + 1):
            T_time += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + k * x[None, :])

        # Add nonlinear product term
        for i in range(1, 5):  # 4 interacting components
            T_res += lambda_res * (rho_n / i) * np.exp(-gamma * time[:, None]) * np.cos(phi * omega * i * time[:, None])

        # Add higher-dimensional term with cross-interaction
        T_extra += higher_dim_factor * np.exp(-gamma * time[:, None]) * np.cos(omega * x[None, :] + k * time[:, None])
        T_extra += lambda_res * higher_dim_factor**2 * np.exp(-gamma * time[:, None])

        # Time and spatial derivatives
        T_time_derivative = np.gradient(T_time + T_res + T_extra, time, axis=0)
        T_space_derivative = np.gradient(T_time + T_res + T_extra, x, axis=1)

        # Total divergence
        divergence = T_time_derivative + T_space_derivative

        # Time-averaged divergence
        time_averaged_divergence = np.mean(np.abs(divergence))
        results.append((lambda_res, higher_dim_factor, time_averaged_divergence))

        # Plot divergence map
        plt.figure(figsize=(10, 6))
        plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', origin='lower', cmap='coolwarm')
        plt.colorbar(label="Divergence")
        plt.contour(divergence, levels=10, colors='k', linewidths=0.5)
        plt.title(f"Divergence Map ($\\lambda_{{res}}$={lambda_res}, Higher-Dim Factor={higher_dim_factor})")
        plt.xlabel("Space (x)")
        plt.ylabel("Time (t)")
        plt.show()

# Aggregate results for plotting
lambda_res_values, higher_dim_factors, divergences = zip(*results)
lambda_res_unique = np.unique(lambda_res_values)
higher_dim_unique = np.unique(higher_dim_factors)

# Plot time-averaged divergence
plt.figure(figsize=(10, 6))
for factor in higher_dim_unique:
    avg_divergence = [d for l, f, d in results if f == factor]
    plt.plot(lambda_res_unique, avg_divergence, marker='o', label=f"Higher-Dim Factor = {factor:.2f}")

plt.title("Mean Absolute Time-Averaged Divergence (Refined Parameters)")
plt.xlabel("Nonlinear Coupling Strength ($\\lambda_{res}$)")
plt.ylabel("Mean Absolute Time-Averaged Divergence")
plt.grid()
plt.legend()
plt.show()