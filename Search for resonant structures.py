import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Settings ---
# Time and Space Parameters
time_start, time_end, time_steps = 0, 50, 500
space_start, space_end, space_steps = 0, 10, 50

# Dimensions for extended simulation
z_max, w_max = 5, 5  # Additional spatial dimensions

# Physical Parameters
rho_c = 1.0  # Base energy density
n_max = 5  # Number of harmonic modes
lambda_res = 0.5  # Non-linear contribution
omega_base = 1.0  # Base angular frequency
gamma_base = 0.05  # Base damping factor
k_x = 0.3  # Wave vector
extra_dim_contribution = 0.1  # Higher-dimensional contribution factor

# Generate Time and Space Arrays
time_array = np.linspace(time_start, time_end, time_steps)
space_array = np.linspace(space_start, space_end, space_steps)

# --- Function Definitions ---
def harmonic_wave(x, t, n_max, omega_base, k_x):
    """Compute harmonic wave contributions."""
    n = np.arange(1, n_max + 1).reshape(-1, 1, 1)  # Reshape for broadcasting
    wave = np.sum((1 / n) * np.sin(n * omega_base * t + k_x * x), axis=0)
    return wave

def nonlinear_term(t, lambda_res, omega_base, gamma_base):
    """Compute non-linear resonant term."""
    return lambda_res * np.cos(omega_base * t) * np.exp(-gamma_base * t)

def higher_dimensional_term(t_array, z_max, w_max, omega_yz, k_yz, extra_dim_contribution):
    """Compute higher-dimensional contributions."""
    z_vals = np.linspace(0, z_max, 50)
    w_vals = np.linspace(0, w_max, 50)
    Z, W = np.meshgrid(z_vals, w_vals)

    # Compute higher-dimensional contributions
    term = np.cos(omega_yz * Z + k_yz * t_array[:, None, None]) * np.sin(omega_yz * W) * extra_dim_contribution

    # Sum along extra dimensions and normalize
    return np.sum(term, axis=(1, 2)) / (len(z_vals) * len(w_vals))

def combined_structure_5d(x, t, z_max, w_max, rho_c, n_max, lambda_res, extra_dim_contribution):
    """Compute combined contributions including 5D terms."""
    harmonic_component = harmonic_wave(x, t, n_max, omega_base, k_x)
    nonlinear_component = nonlinear_term(t, lambda_res, omega_base, gamma_base)
    higher_dim_component = higher_dimensional_term(t, z_max, w_max, 0.4, 0.2, extra_dim_contribution).reshape(t.shape)

    return rho_c + harmonic_component + nonlinear_component + higher_dim_component

def visualize_structure_3d(space_array, time_array):
    """Visualize the combined structure in 3D."""
    X, T = np.meshgrid(space_array, time_array)
    results = combined_structure_5d(X, T, z_max, w_max, rho_c, n_max, lambda_res, extra_dim_contribution)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, results, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Combined Contribution')
    ax.set_title('3D Visualization of Combined Structure')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print(f"space_array shape: {space_array.shape}, time_array shape: {time_array.shape}")
    visualize_structure_3d(space_array, time_array)
