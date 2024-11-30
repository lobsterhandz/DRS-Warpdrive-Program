import numpy as np
import matplotlib.pyplot as plt

# Constants
omega = 2.0  # Base frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
k = 1.0  # Spatial wave number
rho_n = 1.0  # Base energy density for n = 1
lambda_res = 0.5  # Coupling strength for nonlinear terms
time = np.linspace(0, 10, 500)  # Time array
x = np.linspace(0, 10, 500)  # Space array
higher_dim_factor = 0.1  # Scaling for higher-dimensional terms

# Test parameters
gamma_values = [0.1, 0.2, 0.3, 0.4]  # Damping factors to test
n_max_values = [10, 20, 30]  # Number of terms in summation

# Initialize storage for results
convergence_results = []
damping_results = []

# Loop through damping factors and max terms
for gamma in gamma_values:
    for n_max in n_max_values:
        # Initialize components
        T_time = np.zeros((len(time), len(x)))
        T_space = np.zeros((len(time), len(x)))
        T_res = np.zeros((len(time), len(x)))
        T_extra = np.zeros((len(time), len(x)))

        # Compute the damped oscillatory term
        for n in range(1, n_max + 1):
            T_time += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + k * x[None, :])
            T_space += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + k * x[None, :])

        # Add nonlinear product term
        for i in range(1, 5):  # 4 interacting components
            T_res += lambda_res * (rho_n / i) * np.exp(-gamma * time[:, None]) * np.cos(phi * omega * i * time[:, None])

        # Add higher-dimensional term (compactified)
        T_extra += higher_dim_factor * np.exp(-gamma * time[:, None]) * np.cos(omega * x[None, :] + k * time[:, None])

        # Time and spatial derivatives
        T_time_derivative = np.gradient(T_time + T_res + T_extra, time, axis=0)  # Derivative with respect to time
        T_space_derivative = np.gradient(T_time + T_res + T_extra, x, axis=1)  # Derivative with respect to space

        # Total divergence
        divergence = T_time_derivative + T_space_derivative

        # Time-averaged divergence
        time_averaged_divergence = np.mean(divergence, axis=0)
        damping_results.append((gamma, n_max, np.mean(np.abs(time_averaged_divergence))))

        # Store convergence data for this gamma
        if gamma == gamma_values[0]:  # Only do convergence analysis once per n_max
            convergence_results.append((n_max, np.mean(np.abs(time_averaged_divergence))))

# Plot 1: Convergence analysis
plt.figure(figsize=(10, 6))
for n_max, result in convergence_results:
    plt.plot(n_max, result, marker='o', label=f"n_max = {n_max}")
plt.title("Convergence of Oscillatory Series")
plt.xlabel("Number of Terms (n_max)")
plt.ylabel("Mean Absolute Time-Averaged Divergence")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Damping factor sensitivity
plt.figure(figsize=(10, 6))
for gamma, n_max, result in damping_results:
    plt.scatter(gamma, result, label=f"gamma = {gamma}, n_max = {n_max}")
plt.title("Sensitivity to Damping Factor")
plt.xlabel("Damping Factor ($\\gamma$)")
plt.ylabel("Mean Absolute Time-Averaged Divergence")
plt.grid()
plt.show()
