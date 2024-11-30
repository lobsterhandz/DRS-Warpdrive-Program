import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.2  # Optimal damping factor (from sensitivity analysis)
omega = 2.0  # Base frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
k = 1.0  # Spatial wave number
rho_n = 1.0  # Base energy density for n = 1
time = np.linspace(0, 10, 500)  # Time array
x = np.linspace(0, 10, 500)  # Space array
higher_dim_factor = 0.1  # Scaling for higher-dimensional terms

# Test parameters for nonlinear term
lambda_res_values = [0.5, 1.0, 1.5]  # Nonlinear coupling strengths
n_max = 20  # Number of terms in the summation (sufficient for convergence)

# Initialize storage for results
nonlinear_results = []

# Loop through nonlinear coupling strengths
for lambda_res in lambda_res_values:
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
    nonlinear_results.append((lambda_res, np.mean(np.abs(time_averaged_divergence))))

    # Plot divergence map for each lambda_res
    plt.figure(figsize=(10, 6))
    plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(label="Divergence")
    plt.title(f"Divergence of $T_{{\\mu\\nu}}$ (Nonlinear Term $\lambda_{{res}} = {lambda_res}$)")
    plt.xlabel("Space (x)")
    plt.ylabel("Time (t)")
    plt.show()

# Plot: Nonlinear coupling sensitivity
lambda_values, mean_divergences = zip(*nonlinear_results)
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, mean_divergences, marker='o', label="Nonlinear Coupling Sensitivity")
plt.title("Sensitivity of Mean Divergence to Nonlinear Coupling Strength")
plt.xlabel("Nonlinear Coupling Strength ($\lambda_{res}$)")
plt.ylabel("Mean Absolute Time-Averaged Divergence")
plt.grid()
plt.legend()
plt.show()
