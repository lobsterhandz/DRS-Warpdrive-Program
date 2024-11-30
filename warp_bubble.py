from manim import *
import numpy as np
import os
os.environ["FFMPEG_BINARY"] = "ffmpeg.exe"



# Constants for spacetime dynamics
AMPLITUDE = 2.0     # Initial amplitude of the wave
VELOCITY = 0.1      # Speed of the craft
WAVELENGTH = 5.0    # Spacetime wave's periodicity
DAMPING = 0.05      # Damping factor
TORSION_EFFECT = 1.5 # Torsion squeeze effect
GRID_SIZE = 30      # Resolution of the grid

class TorsionDrivenWarpBubble(ThreeDScene):
    def construct(self):
        # Set up 3D axes
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-3, 3, 1],
            axis_config={"color": GREY},
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)

        # Initialize spacetime grid
        grid = Surface(
            lambda u, v: axes.c2p(u, v, self.calculate_bubble(u, v, 0)),
            u_range=[-10, 10],
            v_range=[-10, 10],
            resolution=(GRID_SIZE, GRID_SIZE),
            fill_opacity=0.8,
            color=BLUE_E,
        )
        self.add(grid)

        # Create the craft (a sphere)
        craft = Sphere(
            center=axes.c2p(-10, 0, 0),
            radius=0.5,
            color=RED,
            gloss=0.5,
        )
        self.add(craft)

        # Animation
        self.play(
            UpdateFromAlphaFunc(
                grid,
                lambda mob, alpha: mob.become(
                    Surface(
                        lambda u, v: axes.c2p(
                            u, v, self.calculate_bubble(u, v, alpha * 100)
                        ),
                        u_range=[-10, 10],
                        v_range=[-10, 10],
                        resolution=(GRID_SIZE, GRID_SIZE),
                        fill_opacity=0.8,
                        color=BLUE_E,
                    )
                ),
            ),
            UpdateFromAlphaFunc(
                craft,
                lambda mob, alpha: mob.move_to(
                    axes.c2p(-10 + alpha * VELOCITY * 100, 0, 0)
                ),
            ),
            run_time=10,
            rate_func=linear,
        )

def calculate_bubble(self, x, y, t):
    """
    Generate spacetime deformation (bubble) caused by torsion and wave dynamics.
    """
    # Distance from the craft's current position
    x_c = -10 + VELOCITY * t  # Craft's x-coordinate
    r = np.sqrt((x - x_c) ** 2 + y**2)

    # Torsion squeeze effect on one side of the grid
    torsion = TORSION_EFFECT * np.exp(-(x + 10) ** 2)  # Focused on x = -10 initially

    # Wave-like behavior with damping
    bubble = (
        AMPLITUDE * np.exp(-r**2 / 2) * np.cos(2 * np.pi * r / WAVELENGTH - t)
    )

    # Stabilizing damping factor over time
    damping = np.exp(-DAMPING * t)

    return (bubble + torsion) * damping
