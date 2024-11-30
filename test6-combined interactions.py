import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.2  # Damping factor
omega = 2.0  # Base frequency
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
k = 1.0  # Spatial wave number
rho_n = 1.0  # Base energy density
time = np.linspace(0, 10, 500)  # Time array
x = np.linspace(0, 10, 500)  # Space array

# Parameter ranges
lambda_res_values = [0.5, 0.75, 1.0]  # Nonlinear coupling strengths
higher_dim_factors = [0.1, 0.2]  # Higher-dimensional scaling factors
n_max = 20  # Number of terms in the summation

# Store results
interaction_results = []

# Loop through parameter combinations
for lambda_res in lambda_res_values:
    for higher_dim_factor in higher_dim_factors:
        # Initialize components
        T_time = np.zeros((len(time), len(x)))
        T_res = np.zeros((len(time), len(x)))
        T_extra = np.zeros((len(time), len(x)))

        # Compute damped oscillatory term
        for n in range(1, n_max + 1):
            T_time += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + k * x[None, :])

        # Add nonlinear product term
        for i in range(1, 5):
            T_res += lambda_res * (rho_n / i) * np.exp(-gamma * time[:, None]) * np.cos(phi * omega * i * time[:, None])

        # Add higher-dimensional term
        T_extra += higher_dim_factor * np.exp(-gamma * time[:, None]) * np.cos(omega * x[None, :] + k * time[:, None])

        # Total divergence
        T_total = T_time + T_res + T_extra
        T_time_derivative = np.gradient(T_total, time, axis=0)  # Derivative with respect to time
        T_space_derivative = np.gradient(T_total, x, axis=1)  # Derivative with respect to space
        divergence = T_time_derivative + T_space_derivative

        # Time-averaged divergence
        time_averaged_divergence = np.mean(divergence, axis=0)
        interaction_results.append((lambda_res, higher_dim_factor, np.mean(np.abs(time_averaged_divergence))))

        # Plot divergence map
        plt.figure(figsize=(10, 6))
        plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', origin='lower', cmap='coolwarm')
        plt.colorbar(label="Divergence")
        plt.title(f"Divergence Map ($\lambda_{{res}}$ = {lambda_res}, Higher-Dim Factor = {higher_dim_factor})")
        plt.xlabel("Space (x)")
        plt.ylabel("Time (t)")
        plt.show()

# Plot: Interaction sensitivity
interaction_data = np.array(interaction_results)
for higher_dim_factor in higher_dim_factors:
    mask = interaction_data[:, 1] == higher_dim_factor
    plt.plot(interaction_data[mask, 0], interaction_data[mask, 2], marker='o', label=f"Higher-Dim Factor = {higher_dim_factor}")

# Correct the title and labels by using raw strings (prefix `r`) or escaping backslashes
plt.title("Divergence Map ($\\lambda_{\\text{res}}$ = " + f"{lambda_res}, Higher-Dim Factor = {higher_dim_factor})")
plt.xlabel("Nonlinear Coupling Strength ($\\lambda_{\\text{res}}$)")
plt.ylabel("Mean Absolute Time-Averaged Divergence")
plt.legend()
plt.grid()
plt.show()
