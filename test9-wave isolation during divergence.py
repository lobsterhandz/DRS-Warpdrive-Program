import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.2  # Damping factor
omega = 2.0  # Oscillatory frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
rho_n = 1.0  # Energy density
time = np.linspace(0, 10, 1000)
x = np.linspace(0, 10, 1000)

# Parameters
lambda_res_values = np.linspace(0.5, 0.8, 4)
higher_dim_factors = np.linspace(0.1, 0.3, 4)
n_max = 20

# Enhanced Analysis
for lambda_res in lambda_res_values:
    for higher_dim_factor in higher_dim_factors:
        # Components
        T_time = np.zeros((len(time), len(x)))
        T_res = np.zeros((len(time), len(x)))
        T_extra = np.zeros((len(time), len(x)))

        # Damped oscillations
        for n in range(1, n_max + 1):
            T_time += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + x[None, :])

        # Nonlinear term
        T_res += lambda_res * np.exp(-gamma * time[:, None]) * np.cos(phi * omega * time[:, None] + x[None, :])

        # Higher-dimensional contribution
        T_extra += higher_dim_factor * np.exp(-gamma * time[:, None]) * np.cos(omega * x[None, :] + time[:, None])

        # Total divergence
        divergence = np.gradient(T_time + T_res + T_extra, time, axis=0) + np.gradient(T_time + T_res + T_extra, x, axis=1)

        # Plot divergence
        plt.figure(figsize=(10, 6))
        plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', cmap='coolwarm')
        plt.colorbar(label="Divergence")
        plt.contour(divergence, levels=10, colors='k', linewidths=0.5)
        plt.title(f"Divergence ($\\lambda_{{res}}$={lambda_res}, Higher-Dim Factor={higher_dim_factor})")
        plt.xlabel("Space (x)")
        plt.ylabel("Time (t)")
        plt.show()
