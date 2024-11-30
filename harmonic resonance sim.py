import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Settings ---
# Time parameters
time_start = 0
time_end = 50
time_steps = 1000

# Parameters for the components of the energy tensor
rho_c = 1.0    # Base energy density constant
n_max = 5      # Number of oscillatory modes to consider
lambda_res = 0.5  # Non-linear product contribution
gamma_base = 0.05  # Base damping factor
gamma_variation = 0.02  # Variation in damping factor for modulation
omega_mod = 0.1  # Frequency of damping modulation
omega_base = 1.0   # Base frequency of oscillations
k_x = 0.3  # Wave vector factor
coupling_base = 0.1  # Base coupling strength
coupling_modulation = 0.05  # Variation in coupling strength for modulation
coupling_mod_freq = 0.05  # Frequency for modulating the coupling strength
golden_ratio = (1 + np.sqrt(5)) / 2  # The golden ratio
core_resonance_frequency = 0.2  # Core resonant frequency to synchronize components
observer_focus = 0.05  # Observer's influence factor for focusing the system
observer_state = 0.5  # Probability that the observer is influencing the system (0 to 1)

# Time array for the simulation
time_array = np.linspace(time_start, time_end, time_steps)

# --- Function Definitions ---

def harmonic_coupling(t, n_max, omega_base, gamma, coupling_strength, observer_focus):
    # Sum up oscillatory modes to represent harmonic contributions, using the golden ratio
    harmonic_sum = 0.0
    for n in range(1, n_max + 1):
        phase_adjustment = observer_focus * np.sin(t * core_resonance_frequency)  # Observer's influence on phase
        harmonic_sum += (1 / n**2) * np.cos(n * omega_base * (golden_ratio % n) * t + k_x + coupling_strength * np.sin(omega_base * t) + phase_adjustment)
    return harmonic_sum


def nonlinear_term(t, lambda_res, omega_i_list, gamma, observer_focus):
    # Product term considering multiple rho values with observer influence
    product_term = lambda_res * np.prod([
        np.cos(omega_i * t + observer_focus * np.cos(core_resonance_frequency * t)) for omega_i in omega_i_list
    ]) * np.exp(-gamma * t)
    return product_term


def higher_dimensional_term(t, y_max, omega_y, k_y):
    # Represent the higher-dimensional integral by a sum over compactified dimensions
    integral_sum = 0.0
    y_vals = np.linspace(0, y_max, 20)
    for y in y_vals:
        integral_sum += np.cos(omega_y * y + k_y * t)
    return integral_sum / len(y_vals)


def modulated_damping(t, gamma_base, gamma_variation, omega_mod):
    # Damping function that modulates over time
    return gamma_base + gamma_variation * np.sin(omega_mod * t)


def modulated_coupling(t, coupling_base, coupling_modulation, coupling_mod_freq):
    # Coupling strength that modulates over time
    return coupling_base + coupling_modulation * np.sin(coupling_mod_freq * t)


def observer_effect(t, observer_state):
    # Function to modulate between wave and particle behavior based on observer influence
    # Sigmoid function to simulate the observer effect
    return 1 / (1 + np.exp(-10 * (observer_state - 0.5)))  # Sharp transition around observer_state = 0.5

# --- Main Function for Running the Simulation ---
def beautiful_equation(t, rho_c, n_max, lambda_res, gamma_base, gamma_variation, omega_mod, omega_base, k_x, coupling_base, coupling_modulation, coupling_mod_freq, core_resonance_frequency, observer_focus, observer_state):
    omega_i_list = [omega_base * (golden_ratio ** i) for i in range(4)]  # Frequencies for the nonlinear term, using golden ratios
    omega_y, k_y = 0.4, 0.2  # Settings for higher-dimensional contribution
    y_max = 5

    # Modulated damping term
    gamma = modulated_damping(t, gamma_base, gamma_variation, omega_mod)

    # Modulated coupling term
    coupling_strength = modulated_coupling(t, coupling_base, coupling_modulation, coupling_mod_freq)

    # Observer effect influencing wave vs particle behavior
    observer_influence = observer_effect(t, observer_state)

    # Harmonic contribution with observer's influence on focus
    if observer_influence > 0.5:
        # Particle-like behavior: localized spikes
        harmonic_contribution = harmonic_coupling(t, n_max, omega_base, gamma, coupling_strength, observer_focus) * observer_influence
    else:
        # Wave-like behavior: smooth oscillations
        harmonic_contribution = harmonic_coupling(t, n_max, omega_base, gamma, coupling_strength, observer_focus) * (1 - observer_influence) + np.cos(core_resonance_frequency * t)

    nonlinear_contribution = nonlinear_term(t, lambda_res, omega_i_list, gamma, observer_focus)
    higher_dim_contribution = higher_dimensional_term(t, y_max, omega_y, k_y)

    # Summing up all contributions
    result = rho_c + harmonic_contribution + nonlinear_contribution + higher_dim_contribution
    return result


# --- Running the Simulation ---
results = []
for t in time_array:
    result = beautiful_equation(t, rho_c, n_max, lambda_res, gamma_base, gamma_variation, omega_mod, omega_base, k_x, coupling_base, coupling_modulation, coupling_mod_freq, core_resonance_frequency, observer_focus, observer_state)
    results.append(result)

# --- Visualization ---
plt.figure(figsize=(14, 10))

# Plot 1: Energy Tensor Dynamics over Time
plt.subplot(2, 1, 1)
plt.plot(time_array, results, label='Energy Tensor Dynamics')
plt.xlabel('Time')
plt.ylabel('Energy Contribution')
plt.title('Harmonic Coupling: A Weave of Resonance Over Time')
plt.legend()
plt.grid(True)

# Plot 2: Phase-Space Dynamics to Visualize Infinity Symbol
plt.subplot(2, 1, 2)
plt.plot(results, np.gradient(results, time_array), label='Phase-Space Dynamics')
plt.xlabel('Energy Contribution')
plt.ylabel('Rate of Change of Energy')
plt.title('Phase-Space Plot: Revealing the Weave of Infinity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Further Steps ---
# To visualize the wave-particle duality effect:
# Added observer influence to transition between wave-like and particle-like behaviors.
# Introduced a sigmoid function to create a sharp transition based on the observer's influence.
# The system collapses to localized behavior (particle) when observed and remains wave-like otherwise.
