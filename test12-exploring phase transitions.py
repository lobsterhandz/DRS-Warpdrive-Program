import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light
rho_0 = 1e-9  # Base energy density
gamma = 0.5  # Decay constant
k = 2 * np.pi / 10  # Wavenumber
omega_base = 2 * np.pi * 0.1  # Base frequency
x = np.linspace(0, 10, 500)
t = np.linspace(0, 10, 500)

# Parameters to vary
lambdas = np.linspace(0.1, 1.0, 5)  # Range of lambda values
omegas = np.linspace(0.1, 2.0, 5)  # Range of omega values

# Initialize storage for entropy and variance
entropy_results = []
variance_results = []

# Function to calculate T_mu_nu
def calculate_stress_energy(lmbda, omega):
    T_mu_nu = rho_0 * c**2
    for n in range(1, 5):  # Summing over harmonic terms
        T_mu_nu += (rho_0 / n**2) * np.exp(-gamma * t[:, None]) * (
            1 + np.cos(n * omega * t[:, None] + k * x[None, :])
        )
    T_mu_nu += (
        lmbda
        * np.prod([rho_0 * np.cos(omega * t[:, None]) for _ in range(4)], axis=0)
    )
    return T_mu_nu

# Loop over parameters and analyze behavior
for lmbda in lambdas:
    for omega in omegas:
        T_mu_nu = calculate_stress_energy(lmbda, omega)
        
        # Calculate entropy (as a proxy for disorder)
        probability_density = np.abs(T_mu_nu) / np.sum(np.abs(T_mu_nu))
        entropy = -np.sum(probability_density * np.log(probability_density + 1e-10))
        entropy_results.append((lmbda, omega, entropy))
        
        # Calculate variance (as a measure of amplitude spread)
        variance = np.var(T_mu_nu)
        variance_results.append((lmbda, omega, variance))

# Convert results to numpy arrays for plotting
entropy_results = np.array(entropy_results)
variance_results = np.array(variance_results)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(entropy_results[:, 0], entropy_results[:, 2], c=entropy_results[:, 1], cmap='viridis')
plt.colorbar(label='Frequency (omega)')
plt.xlabel('Nonlinear Coupling Strength (lambda)')
plt.ylabel('Entropy')
plt.title('Phase Transitions: Entropy vs Lambda')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(variance_results[:, 0], variance_results[:, 2], c=variance_results[:, 1], cmap='plasma')
plt.colorbar(label='Frequency (omega)')
plt.xlabel('Nonlinear Coupling Strength (lambda)')
plt.ylabel('Variance')
plt.title('Phase Transitions: Variance vs Lambda')
plt.show()
