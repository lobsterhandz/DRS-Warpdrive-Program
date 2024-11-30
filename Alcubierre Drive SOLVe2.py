import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Constants for spacetime grid
GRID_SIZE = 50
TIME_STEPS = 200
VELOCITY = 0.05  # Speed of warp bubble movement
DAMPING = 0.1    # Damping factor for resonance stability
AMPLITUDE = 1.0  # Amplitude of spacetime curvature

# Unified Resonance Equation Terms
def T_osc(t, x, y, frequencies, amplitudes):
    """Oscillatory tensor to model harmonic oscillations in spacetime."""
    return np.sum(
        [amp * np.sin(2 * np.pi * freq * t + x + y) for freq, amp in zip(frequencies, amplitudes)], axis=0
    )

def T_res(t, x, y, base_freq, amplitude, damping):
    """Resonance tensor to model nonlinear interactions."""
    return amplitude * np.cos(2 * np.pi * base_freq * t) * np.exp(-damping * t)

def warp_bubble(t, x, y, velocity):
    """Warp field curvature function."""
    r = np.sqrt((x - velocity * t)**2 + y**2)
    return -AMPLITUDE * np.exp(-r**2)

def energy_density(x, y, t):
    """Approximate stress-energy tensor components for visualization."""
    curvature = warp_bubble(t, x, y, VELOCITY)
    return curvature**2 / (1 + np.abs(curvature))  # Simplified metric

def gravitational_wave_emission(t, x, y):
    """Model propagation of gravitational waves from the warp bubble."""
    distance = np.sqrt(x**2 + y**2)
    return np.sin(2 * np.pi * distance - VELOCITY * t) / (1 + distance)

# Grid and time setup
x = np.linspace(-10, 10, GRID_SIZE)
y = np.linspace(-10, 10, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Initialize frequency and amplitude arrays for oscillatory tensor
frequencies = [0.5, 1.0, 1.5, 2.0]
amplitudes = [1.0, 0.8, 0.6, 0.4]

# Data logging setup
data_log = []

# Create figure and axes for plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Advanced Alcubierre Drive with URE and Quantum Resonance Visualization')

# Initialize warp bubble and craft (sphere)
craft_position = [0, 0]
craft, = ax.plot([craft_position[0]], [craft_position[1]], [0], 'ro', markersize=10)

# Animation function
def update(frame):
    t = frame / TIME_STEPS
    # Warp bubble curvature
    Z = warp_bubble(t, X, Y, VELOCITY)
    # Add oscillatory and resonance tensors
    Z_osc = T_osc(t, X, Y, frequencies, amplitudes) * 0.2
    Z_res = T_res(t, X, Y, base_freq=0.5, amplitude=0.5, damping=DAMPING)
    Z += Z_osc + Z_res

    # Energy density visualization
    energy = energy_density(X, Y, t)
    gravitational_waves = gravitational_wave_emission(t, X, Y)

    # Calculate stability metrics
    max_curvature = np.max(Z)
    min_curvature = np.min(Z)
    avg_curvature = np.mean(Z)
    flatness = np.std(Z)

    # Log data
    data_log.append({
        "Time": t,
        "Max Curvature": max_curvature,
        "Min Curvature": min_curvature,
        "Average Curvature": avg_curvature,
        "Flatness": flatness
    })

    # Update craft position
    craft_position[0] = VELOCITY * t
    craft.set_data([craft_position[0]], [craft_position[1]])
    craft.set_3d_properties([0])

    # Clear and re-plot surface
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Advanced Alcubierre Drive with URE and Quantum Resonance Visualization')

    # Plot warp bubble curvature
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.7, cmap='viridis')

    # Overlay energy density
    ax.plot_surface(X, Y, energy, rstride=1, cstride=1, alpha=0.5, cmap='plasma')

    # Overlay gravitational wave emissions
    ax.plot_surface(X, Y, gravitational_waves, rstride=1, cstride=1, alpha=0.3, cmap='cool')

    # Plot craft
    ax.plot([craft_position[0]], [craft_position[1]], [0], 'ro', markersize=10)
    return ax,

# Animate the warp drive visualization
ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50, blit=False)

# Show the animation
plt.show()

# Save logged data to CSV
df = pd.DataFrame(data_log)
df.to_csv("refined_warp_bubble_data.csv", index=False)
print("Data saved to 'refined_warp_bubble_data.csv'.")

# Output summary statistics
print("Warp Bubble Data Summary:")
print(df.describe())
