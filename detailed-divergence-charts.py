import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.3  # Increased damping factor to suppress residual oscillations
omega = 2.0  # Base frequency
k = 1.0      # Spatial wave number
rho_n = 1.0  # Base energy density for n = 1
n_max = 20   # Increased number of terms in the summation
time_steps = 1000  # Higher temporal resolution
space_steps = 1000  # Higher spatial resolution
time = np.linspace(0, 10, time_steps)  # Time array
x = np.linspace(0, 10, space_steps)    # Space array

# Initialize T_mu_nu components as 2D arrays
T_time = np.zeros((len(time), len(x)))  # Position-dependent energy density
T_space = np.zeros((len(time), len(x)))  # Position-dependent energy density

# Compute the damped oscillatory term
for n in range(1, n_max + 1):
    # Oscillatory term with damping
    T_time += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + k * x[None, :])
    T_space += (rho_n / n**2) * np.exp(-gamma * time[:, None]) * np.cos(n * omega * time[:, None] + k * x[None, :])

# Time and spatial derivatives
T_time_derivative = np.gradient(T_time, time, axis=0)  # Derivative with respect to time
T_space_derivative = np.gradient(T_space, x, axis=1)  # Derivative with respect to space

# Total divergence
divergence = T_time_derivative + T_space_derivative

# Time-averaged divergence
time_averaged_divergence = np.mean(divergence, axis=0)

# Plot divergence map
plt.figure(figsize=(10, 6))
plt.imshow(divergence, extent=[x.min(), x.max(), time.min(), time.max()], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label="Divergence")
plt.title("Divergence of $T_{\\mu\\nu}$ Over Time and Space")
plt.xlabel("Space (x)")
plt.ylabel("Time (t)")
plt.show()

# Plot time-averaged divergence
plt.figure(figsize=(10, 6))
plt.plot(x, time_averaged_divergence, label="Time-Averaged Divergence", color="blue")
plt.axhline(0, color="black", linestyle="--", label="Zero Divergence")
plt.title("Time-Averaged Divergence of $T_{\\mu\\nu}$")
plt.xlabel("Space (x)")
plt.ylabel("Divergence (Averaged Over Time)")
plt.legend()
plt.grid()
plt.show()
