import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import fft
from scipy.signal.windows import hamming

# Dynamic Resonance Simulation for Extended Einstein Field Equation
# This simulation will help visualize the behavior of the extended equation, focusing on oscillatory stability, nonlinear resonance, and higher-dimensional interactions.

# Set up constants for oscillatory behavior
A_x = 1.0  # Amplitude A(x), representing initial energy distribution
B_x = 0.8  # Amplitude B(x)
omega_1 = 2.0 * np.pi  # Angular frequency of primary oscillation
omega_2 = 4.0 * np.pi  # Angular frequency of secondary, higher frequency oscillation
k = 1.0  # Wavenumber for spatial variation

gamma_0 = 0.25  # Initial damping factor, in optimal stability range for controlling oscillations
lambda_res = 7.0  # Further amplified nonlinear coupling strength to observe deeper impact
phi = np.pi / 3  # Updated phase constant for richer nonlinear interaction
higher_dim_factor = 1.5  # Increased contribution scaling from higher-dimensional terms for more pronounced influence

# New parameter: stochastic noise factor
noise_factor_initial = 0.1  # Initial stochastic fluctuations to simulate environmental perturbations

# Define the time frame and initial conditions
t_span = (0, 50)  # Time from 0 to 50 units
t_eval = np.linspace(0, 50, 500)  # Evaluation points in time

# Initial conditions for the system
initial_conditions = [A_x, B_x]

# Define the differential equations for oscillatory, nonlinear, and higher-dimensional components

def dynamic_system(t, y):
    A_t, B_t = y  # Unpacking current amplitudes
    
    # Time-dependent damping factor
    gamma = gamma_0 * np.exp(-0.01 * t)
    
    # Time-dependent stochastic noise factor to simulate decreasing noise over time
    noise_factor = noise_factor_initial * np.exp(-0.01 * t)
    
    # Oscillatory contributions with damping
    dA_dt = -gamma * A_t * np.cos(omega_1 * t + k)  # Damped oscillatory effect on A
    dB_dt = -gamma * B_t * np.sin(omega_1 * t + k)  # Damped oscillatory effect on B
    
    # Nonlinear resonant term contributions (amplified)
    nonlinear_effect = lambda_res * np.cos(phi * omega_1 * t) * np.exp(-gamma * t)
    
    # Higher-dimensional contribution (modeled as an additional effect)
    higher_dim_effect = higher_dim_factor * np.cos(omega_2 * t + k * t) * np.exp(-gamma * t)
    
    # Coupling between frequency modes
    frequency_interaction = 0.5 * np.sin(omega_1 * t) * np.cos(omega_2 * t)
    
    # Stochastic noise addition
    stochastic_noise = noise_factor * np.random.normal(0, 1)
    
    # Calculating the changes over time for A_t and B_t
    dA_dt += nonlinear_effect + higher_dim_effect + frequency_interaction + stochastic_noise
    dB_dt += nonlinear_effect - higher_dim_effect + frequency_interaction + stochastic_noise
    
    return [dA_dt, dB_dt]

# Solve the dynamic system using SciPy's solve_ivp
solution = solve_ivp(dynamic_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Check for success in solving
if solution.success:
    print("Solution found successfully!")
else:
    print("Solution failed to converge!")

# Extracting time and solution data
time = solution.t
A_sol, B_sol = solution.y

# Plotting the results
plt.figure(figsize=(18, 20))

# Plot A(t)
plt.subplot(5, 1, 1)
plt.plot(time, A_sol, label=r'$A(t)$ - Oscillatory Component', color='blue')
plt.title("Dynamics of Oscillatory Contributions in Extended Einstein Equation")
plt.xlabel("Time")
plt.ylabel("Amplitude A(t)")
plt.legend()
plt.grid()

# Plot B(t)
plt.subplot(5, 1, 2)
plt.plot(time, B_sol, label=r'$B(t)$ - Oscillatory Component', color='red')
plt.xlabel("Time")
plt.ylabel("Amplitude B(t)")
plt.legend()
plt.grid()

# Plot Nonlinear Contribution
dA_nonlin = lambda_res * np.cos(phi * omega_1 * time) * np.exp(-gamma_0 * time * 0.01)
plt.subplot(5, 1, 3)
plt.plot(time, dA_nonlin, label=r'Nonlinear Contribution', color='green')
plt.xlabel("Time")
plt.ylabel("Nonlinear Effect Amplitude")
plt.legend()
plt.grid()

# Plot Higher-Dimensional Contribution
higher_dim_contrib = higher_dim_factor * np.cos(omega_2 * time + k * time) * np.exp(-gamma_0 * time * 0.01)
plt.subplot(5, 1, 4)
plt.plot(time, higher_dim_contrib, label=r'Higher-Dimensional Contribution', color='purple')
plt.xlabel("Time")
plt.ylabel("Higher-Dimensional Effect Amplitude")
plt.legend()
plt.grid()

# Phase plot of A(t) vs B(t)
plt.subplot(5, 1, 5)
plt.plot(A_sol, B_sol, label=r'Phase Plot $A(t)$ vs $B(t)$', color='orange')
plt.xlabel("A(t)")
plt.ylabel("B(t)")
plt.legend()
plt.grid()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Perform Fourier Transform on A(t) and B(t) to observe frequency distribution
# Apply windowing to reduce spectral leakage
window = hamming(len(A_sol))
A_sol_windowed = A_sol - np.mean(A_sol)  # Remove DC component
B_sol_windowed = B_sol - np.mean(B_sol)  # Remove DC component
A_fft = fft(A_sol_windowed * window)
B_fft = fft(B_sol_windowed * window)
frequencies = np.fft.fftfreq(len(time), d=(time[1] - time[0]))

# Plotting the Fourier Transform results
plt.figure(figsize=(18, 10))

# Plot FFT of A(t)
plt.subplot(2, 1, 1)
plt.plot(frequencies, np.abs(A_fft), label=r'FFT of $A(t)$', color='blue')
plt.title("Frequency Spectrum of A(t) and B(t)")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot FFT of B(t)
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(B_fft), label=r'FFT of $B(t)$', color='red')
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Calculate Lyapunov exponent for chaos detection
def lyapunov_exponent(system, t_span, initial_conditions, epsilon=1e-9, delta_t=0.1):
    t0, tf = t_span
    t_vals = np.arange(t0, tf, delta_t)
    
    y0 = np.array(initial_conditions)
    perturbed_y0 = y0 + epsilon  # Slightly perturbed initial conditions
    
    lyapunov_sum = 0.0
    num_steps = len(t_vals)
    
    for i in range(num_steps - 1):
        sol1 = solve_ivp(system, (t_vals[i], t_vals[i+1]), y0, method='RK45')
        sol2 = solve_ivp(system, (t_vals[i], t_vals[i+1]), perturbed_y0, method='RK45')
        
        if sol1.success and sol2.success:
            dist = np.linalg.norm(sol2.y[:, -1] - sol1.y[:, -1])
            y0 = sol1.y[:, -1]
            perturbed_y0 = sol2.y[:, -1]
            lyapunov_sum += np.log(dist / epsilon)
        else:
            print("Integration failed at step", i)
            return None
    
    lyapunov_exp = lyapunov_sum / (delta_t * num_steps)
    return lyapunov_exp

# Estimate Lyapunov exponent
lyapunov_exp = lyapunov_exponent(dynamic_system, t_span, initial_conditions)
if lyapunov_exp is not None:
    print(f"Estimated Lyapunov Exponent: {lyapunov_exp}")
    if lyapunov_exp > 0:
        print("The system exhibits chaotic behavior.")
    else:
        print("The system does not exhibit chaotic behavior.")
else:
    print("Failed to estimate the Lyapunov exponent.")

# Display summary of insights
print("Insights from the Simulation:\n")
print("1. A(t) and B(t) represent oscillatory dynamics with damping, nonlinear contributions, higher-dimensional interactions, and stochastic fluctuations.")
print("2. The introduction of stochastic noise simulates environmental perturbations, which now decreases over time to reflect transient disturbances.")
print("3. The nonlinear and higher-dimensional contributions introduce significant deviations from the simple oscillatory model, which are visualized in the third and fourth plots.")
print("4. The phase plot of A(t) vs B(t) provides insights into the coupling dynamics, showing complex and potentially chaotic interactions in the system's trajectory in phase space.")
print("5. The frequency spectrum obtained from the Fourier Transform (after applying windowing and removing the DC component) reveals the dominant frequencies in A(t) and B(t), highlighting the influence of multiple oscillatory components while filtering out constant components.")
print("6. The estimated Lyapunov exponent provides insights into the chaotic nature of the system, helping quantify its sensitivity to initial conditions.")
print("7. Stability depends on the balance between damping (gamma), coupling (lambda_res), higher-dimensional effects, stochastic influences, and the interaction between multiple frequency modes.")
