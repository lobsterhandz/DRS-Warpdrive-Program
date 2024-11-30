import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import Planck, Boltzmann, hbar, pi
from scipy.optimize import curve_fit

# Constants
h = Planck
k_B = Boltzmann
c = 3e8  # Speed of light in m/s

# Define functions
def planck_law(frequency, temperature):
    """Planck's law for blackbody radiation."""
    return (2 * h * frequency**3 / c**2) / (np.exp(h * frequency / (k_B * temperature)) - 1)

def quantum_harmonic_oscillator(n, omega):
    """Energy levels of a quantum harmonic oscillator."""
    return hbar * omega * (n + 0.5)

def heat_capacity(T, Debye_temp):
    """Debye model for heat capacity."""
    x = np.linspace(0, Debye_temp / T, 100)
    return 9 * k_B * (T / Debye_temp)**3 * np.trapz((x**4 * np.exp(x)) / (np.exp(x) - 1)**2, x)

# Load additional datasets
quantum_thermal_data = pd.read_csv("quantum_thermal_data.csv")  # Replace with the actual file path

# Simulations
frequencies = np.linspace(1e12, 1e14, 500)  # Example frequency range in Hz
temperatures = [300, 600, 900]  # Example temperatures in Kelvin

# Plot Planck's Law
plt.figure(figsize=(12, 8))
for T in temperatures:
    plt.plot(frequencies, planck_law(frequencies, T), label=f"T={T}K")
plt.title("Blackbody Radiation: Planck's Law")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Intensity")
plt.legend()
plt.grid()
plt.show()

# Plot Quantum Harmonic Oscillator
n_values = np.arange(0, 10)
omega = 1e14  # Example angular frequency
energy_levels = quantum_harmonic_oscillator(n_values, omega)

plt.figure(figsize=(12, 8))
plt.bar(n_values, energy_levels, color='blue', alpha=0.7)
plt.title("Quantum Harmonic Oscillator Energy Levels")
plt.xlabel("Quantum Number (n)")
plt.ylabel("Energy (J)")
plt.grid()
plt.show()

# Plot Heat Capacity
temperatures = np.linspace(1, 300, 300)
Debye_temp = 200  # Example Debye temperature
heat_capacities = [heat_capacity(T, Debye_temp) for T in temperatures]

plt.figure(figsize=(12, 8))
plt.plot(temperatures, heat_capacities, label="Heat Capacity (Debye Model)")
plt.title("Heat Capacity vs Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity (J/K)")
plt.legend()
plt.grid()
plt.show()

# Incorporate into entropy and variance equations
def entropy_eq(x, a, b):
    """Entropy equation (example)."""
    return a * np.log(b * x)

def variance_eq(x, a, b):
    """Variance equation (example)."""
    return a * np.sin(b * x)

# Fit entropy and variance equations
x_data = quantum_thermal_data['x']
y_entropy = quantum_thermal_data['entropy']
y_variance = quantum_thermal_data['variance']

params_entropy, _ = curve_fit(entropy_eq, x_data, y_entropy)
params_variance, _ = curve_fit(variance_eq, x_data, y_variance)

# Plot fitted results
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_entropy, label="Observed Entropy")
plt.plot(x_data, entropy_eq(x_data, *params_entropy), color='red', label="Fitted Entropy")
plt.title("Entropy Fit")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_variance, label="Observed Variance")
plt.plot(x_data, variance_eq(x_data, *params_variance), color='red', label="Fitted Variance")
plt.title("Variance Fit")
plt.legend()
plt.grid()
plt.show()

# Analyze universality across datasets
scaling_results = {
    "Planck Constant Scaling": params_entropy[1] / h,
    "Boltzmann Constant Scaling": params_variance[1] / k_B
}

for key, value in scaling_results.items():
    print(f"{key}: {value}")

# Next steps
print("Next Steps:")
print("1. Validate fits with experimental datasets.")
print("2. Refine equations for universality.")
print("3. Explore higher-dimensional systems for additional insights.")
