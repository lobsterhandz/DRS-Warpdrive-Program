import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Settings ---
# Time and space parameters
space_array = np.linspace(0, 10, 50)
time_array = np.linspace(0, 50, 500)

# Constants and parameters
rho_c = 1.0  # Base energy density constant
n_max = 5  # Number of harmonic modes to consider
lambda_res = 0.5  # Non-linear product contribution
extra_dim_contribution = 0.1  # Contribution from higher dimensions
omega_base = 1.0  # Base angular frequency
k_x = 0.3  # Wave vector factor


# --- Function Definitions ---

def harmonic_wave(x, t, n_max, omega_base, k_x):
    """
    Computes a harmonic wave for a given spatial (x) and temporal (t) input.
    """
    # Convert `x` and `t` to numpy arrays if they're not already
    x = np.asarray(x)
    t = np.asarray(t)

    # Reshape x and t for broadcasting
    if x.ndim == 1:
        x = x[:, None]  # Make x 2D: (space_points, 1)
    if t.ndim == 1:
        t = t[None, :]  # Make t 2D: (1, time_points)

    # Generate harmonic modes
    n = np.arange(1, n_max + 1).reshape(-1, 1, 1)  # Shape (n_modes, 1, 1)

    # Calculate the harmonic wave with proper broadcasting
    wave = np.sum((1 / n) * np.sin(n * omega_base * t + k_x * x), axis=0)  # Summing over modes
    return wave


def combined_structure(x, t, n_max, omega_base, k_x, rho_c, lambda_res, extra_dim_contribution):
    """
    Computes the combined structure by summing multiple components.
    """
    # Calculate harmonic component
    harmonic_component = harmonic_wave(x, t, n_max, omega_base, k_x)

    # Add other contributions (nonlinear, dimensional, etc.)
    nonlinear_component = lambda_res * np.cos(t)
    higher_dim_component = extra_dim_contribution * np.sin(x)

    # Combine components
    combined = rho_c + harmonic_component + nonlinear_component + higher_dim_component
    return combined


def visualize_structure_3d(space_array, time_array, rho_c, n_max, lambda_res, extra_dim_contribution):
    """
    Visualizes the combined structure in 3D.
    """
    X, T = np.meshgrid(space_array, time_array)  # Create a mesh grid
    results = combined_structure(X, T, n_max, omega_base, k_x, rho_c, lambda_res, extra_dim_contribution)

    # Debugging shapes
    print(f"X shape: {X.shape}, T shape: {T.shape}, results shape: {results.shape}")

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, results, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Combined Contribution')
    ax.set_title('3D Visualization of Combined Structure')
    plt.show()


def visualize_cross_section(space_array, time_array, T_index=100):
    """
    Visualizes a 2D cross-section of the combined structure for a fixed time slice.
    """
    X, T = np.meshgrid(space_array, time_array)
    results = combined_structure(X, T, n_max, omega_base, k_x, rho_c, lambda_res, extra_dim_contribution)

    # Select a specific time slice
    cross_section = results[T_index, :]

    # Plot the cross-section
    plt.figure(figsize=(8, 6))
    plt.plot(space_array, cross_section, label=f'Time Index: {T_index}')
    plt.xlabel('Space')
    plt.ylabel('Combined Contribution')
    plt.title('Cross-Section of Combined Structure')
    plt.legend()
    plt.grid()
    plt.show()


def visualize_higher_dim_contributions(space_array, time_array):
    """
    Visualizes higher-dimensional contributions as a heatmap.
    """
    X, Y = np.meshgrid(space_array, space_array)  # Treating another spatial dimension
    results = extra_dim_contribution * np.sin(X + Y)  # Example contribution

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, results, levels=50, cmap='plasma')
    plt.colorbar(label='Contribution Amplitude')
    plt.xlabel('Space (X)')
    plt.ylabel('Space (Y)')
    plt.title('Higher-Dimensional Contributions')
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # Visualize the combined structure in 3D
    visualize_structure_3d(space_array, time_array, rho_c, n_max, lambda_res, extra_dim_contribution)

    # Visualize a cross-section of the combined structure
    visualize_cross_section(space_array, time_array)

    # Visualize higher-dimensional contributions
    visualize_higher_dim_contributions(space_array, time_array)


#evidence of OCillation FOUND!!!