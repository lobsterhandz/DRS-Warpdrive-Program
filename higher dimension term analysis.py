import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.2  # Optimal damping factor
omega = 2.0  # Base frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
k = 1.0  # Spatial wave number
rho_n = 1.0  # Base energy density for n = 1
lambda_res = 0.75  # Optimal nonlinear coupling strength from previous analysis
time = np.linspace(0, 10, 500)  # Time array
x = np.linspace(0, 10, 500)  # Space array

# Higher-dimensional contributions
higher_dim_factors = [0.1, 0.2, 0.3]  # Scaling factors for higher-dimensional terms
n_max = 20  # Number of terms in the summation

# Initialize storage for results
higher_dim_results = []

# Loop through higher-dimensional scaling factors
for higher_dim_factor in higher_dim_factors:
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

    # Add higher-dimensional term
    T_extra += higher_dim_factor * np.exp(-gamma * time[:, None]) * np.cos(omega * x[None, :] + k * time[:, None])

    # Time and spatial derivatives
    T_time_derivative = np.gradient(T_time + T_res + T_extra, time, axis=0)  # Derivative with respect to time
    T_space_derivative = np.gradient(T_time + T_res + T_extra, x, axis=1)  # Derivative with respect to space

    # Total divergence
    divergence = T_time_derivative + T_space_derivative

    # Time-averaged divergence
    time_averaged_divergence = np.mean(divergence, axis=0)
    higher_dim_results.append((higher_dim_factor, np.mean(np.abs(time_averaged_divergence))))

    # Plot divergence map for each higher_dim_factor
    plt.figure(figsize=(10, 6))
    plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(label="Divergence")
    plt.title(f"Divergence of $T_{{\\mu\\nu}}$ (Higher-Dimensional Factor = {higher_dim_factor})")
    plt.xlabel("Space (x)")
    plt.ylabel("Time (t)")
    plt.show()

# Plot: Higher-dimensional factor sensitivity
higher_dim_values, mean_divergences = zip(*higher_dim_results)
plt.figure(figsize=(10, 6))
plt.plot(higher_dim_values, mean_divergences, marker='o', label="Higher-Dimensional Sensitivity")
plt.title("Sensitivity of Mean Divergence to Higher-Dimensional Scaling Factor")
plt.xlabel("Higher-Dimensional Scaling Factor")
plt.ylabel("Mean Absolute Time-Averaged Divergence")
plt.grid()
plt.legend()
plt.show()
