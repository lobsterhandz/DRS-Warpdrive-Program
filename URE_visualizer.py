# Import Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go

# Define URE Parameters
def unified_resonance_equation(time, space, nu_stochastic=0.01, lambda_res=1, gamma_base=0.5):
    """
    Unified Resonance Equation combining oscillatory and resonance tensors.
    
    Parameters:
    - time: Current time step.
    - space: Spatial position.
    - nu_stochastic: Noise factor for stochastic effects.
    - lambda_res: Amplitude scaling for resonance.
    - gamma_base: Decay factor for resonance.

    Returns:
    - Calculated resonance value at the given time and space.
    """
    # Oscillatory tensor component
    oscillation = sum((1 / n) * np.sin(n * np.pi * time + 2 * np.pi * space) for n in range(1, 10))
    
    # Resonance tensor component
    resonance = lambda_res * np.exp(-gamma_base * time) * np.sin(np.pi * time)
    
    # Stochastic noise (optional)
    noise = np.random.normal(0, nu_stochastic)
    
    return oscillation + resonance + noise

# Generate Simulation Data
def simulate_unified_resonance(time_steps, spatial_points):
    """
    Simulates the Unified Resonance Equation over a grid of time and space.
    
    Parameters:
    - time_steps: Number of discrete time steps.
    - spatial_points: Number of discrete spatial points.

    Returns:
    - A 2D NumPy array with calculated values for each (time, space) pair.
    """
    time = np.linspace(0, 10, time_steps)
    space = np.linspace(0, 10, spatial_points)
    results = []

    for t_val in time:
        row = [unified_resonance_equation(t_val, x_val) for x_val in space]
        results.append(row)
    
    return np.array(results)

def animate_results(results, spatial_points):
    """
    Creates an animated visualization of the simulation data.
    
    Parameters:
    - results: The simulation data as a 2D array, where each row is a spatial slice at a time step.
    - spatial_points: Number of spatial points used in the simulation.
    """
    # Ensure the results are reshaped into 2D
    results_reshaped = np.reshape(results, (len(results), spatial_points))
    
    fig, ax = plt.subplots()
    cax = ax.imshow(results_reshaped[0].reshape(1, -1), 
                    extent=[0, spatial_points, 0, len(results)], 
                    aspect='auto', cmap='viridis')
    fig.colorbar(cax)

    def update(frame):
        # Update the data for each frame
        cax.set_array(results_reshaped[frame].reshape(1, -1))
        return cax,

    ani = animation.FuncAnimation(fig, update, frames=len(results_reshaped), blit=False)
    plt.show()

# Export to Interactive Visualization
def export_visualization(results):
    """
    Exports the simulation results as an interactive 3D HTML visualization.
    
    Parameters:
    - results: The simulation data as a 2D array.
    """
    fig = go.Figure(data=go.Surface(z=results))
    fig.update_layout(
        title='Unified Resonance Equation Visualization',
        scene=dict(
            xaxis_title='Space',
            yaxis_title='Time',
            zaxis_title='Amplitude'
        )
    )
    fig.write_html("URE_visualization.html")
    print("Visualization exported to URE_visualization.html")

# Main Execution
if __name__ == "__main__":
    # Simulation settings
    time_steps = 100
    spatial_points = 50

    print("Simulating Unified Resonance Equation...")
    data = simulate_unified_resonance(time_steps, spatial_points)
    print("Simulation complete.")

    # Animate results
    print("Creating animation...")
    animate_results(data, spatial_points)

    # Export results to interactive HTML
    print("Exporting visualization to HTML...")
    export_visualization(data)
    print("Done!")
