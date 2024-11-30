import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)


# Constants
GRID_SIZE = 20  # 20x20x20 grid
TIME_STEPS = 200
CRAFT_RADIUS = 0.5  # Radius of the spherical craft
VELOCITY = 0.05  # Initial warp bubble speed
AMPLITUDE = 1.0  # Warp bubble curvature strength
DAMPING = 0.1  # Damping factor for stabilization

# Create 3D grid points
x = np.linspace(-10, 10, GRID_SIZE)
y = np.linspace(-10, 10, GRID_SIZE)
z = np.linspace(-10, 10, GRID_SIZE)
X, Y, Z = np.meshgrid(x, y, z)

# Flatten grid for particle calculations
points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)

# Craft position (stationary)
craft_position = np.array([0, 0, 0])

# Warp bubble equations
def warp_bubble(t, point, velocity, amplitude, damping):
    r = np.linalg.norm(point - craft_position - np.array([velocity * t, 0, 0]))
    warp_effect = -amplitude * np.exp(-r**2)
    resonance_effect = amplitude * np.cos(2 * np.pi * 0.5 * t) * np.exp(-damping * t)
    return warp_effect + resonance_effect

# Plot setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Dynamic Warp Bubble Visualization')

# Initial plot
craft = ax.scatter(*craft_position, color='red', s=100, label='Craft (Sphere)')
grid = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=5, alpha=0.7, label='Spacetime Points')

# Animation function
def update(frame):
    t = frame / TIME_STEPS
    new_points = []
    for point in points:
        warp_effect = warp_bubble(t, point, VELOCITY, AMPLITUDE, DAMPING)
        new_point = point + warp_effect * (point - craft_position) * 0.01
        new_points.append(new_point)
    new_points = np.array(new_points)
    grid._offsets3d = (new_points[:, 0], new_points[:, 1], new_points[:, 2])
    ax.set_title(f"Dynamic Warp Bubble - Frame {frame}")
    return grid,

# Animate the plot
ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50, blit=False)



plt.show()
ani.save('warp_bubble_visualization.mp4', writer=r'C:\Users\----\OneDrive\Documents\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe')
