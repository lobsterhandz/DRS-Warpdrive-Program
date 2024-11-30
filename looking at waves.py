import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants for simulation
SPATIAL_POINTS = 100  # Number of points in space
TIME_STEPS = 200  # Number of time steps in the animation
AMPLITUDE = 2.0  # Base amplitude of waves
FREQUENCY = 0.1  # Frequency of oscillation
RESONANCE_FACTOR = 0.5  # Resonance amplification
NOISE_LEVEL = 0.05  # Stochastic noise intensity

# Function to generate the multi-layered data
def generate_holographic_data(spatial_points, time_steps):
    x = np.linspace(0, 10, spatial_points)  # Spatial dimension
    t = np.linspace(0, 10, time_steps)  # Time dimension
    surface_layer = []  # The observable phenomenon
    hidden_layer = []  # The underlying mechanics

    for time in t:
        # Surface layer: Observable wave patterns
        wave1 = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * x + time)
        wave2 = AMPLITUDE * np.cos(2 * np.pi * FREQUENCY * (x - 2) - time)
        surface = wave1 + wave2

        # Hidden layer: Resonance and stochastic noise
        resonance = RESONANCE_FACTOR * np.sin(4 * np.pi * FREQUENCY * x - 0.5 * time)
        stochastic = NOISE_LEVEL * np.random.normal(0, 1, spatial_points)
        hidden = resonance + stochastic

        # Store both layers
        surface_layer.append(surface)
        hidden_layer.append(hidden)

    return np.array(surface_layer), np.array(hidden_layer)

# Animation function to display layers
def animate_holography(surface_data, hidden_data, spatial_points):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # First subplot: Surface dynamics
    ax1 = axs[0]
    surface_plot = ax1.imshow(surface_data[0].reshape(1, -1), extent=[0, spatial_points, 0, 1], 
                              aspect='auto', cmap='plasma')
    ax1.set_title("Surface Dynamics (Observable)")
    fig.colorbar(surface_plot, ax=ax1)

    # Second subplot: Hidden mechanics
    ax2 = axs[1]
    hidden_plot = ax2.imshow(hidden_data[0].reshape(1, -1), extent=[0, spatial_points, 0, 1], 
                             aspect='auto', cmap='viridis')
    ax2.set_title("Hidden Layer (Underlying Forces)")
    fig.colorbar(hidden_plot, ax=ax2)

    # Animation update function
    def update(frame):
        surface_plot.set_array(surface_data[frame].reshape(1, -1))
        hidden_plot.set_array(hidden_data[frame].reshape(1, -1))
        return surface_plot, hidden_plot

    ani = animation.FuncAnimation(fig, update, frames=len(surface_data), blit=False)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Generating holographic data...")
    surface_data, hidden_data = generate_holographic_data(SPATIAL_POINTS, TIME_STEPS)

    print("Animating holographic visualization...")
    animate_holography(surface_data, hidden_data, SPATIAL_POINTS)
