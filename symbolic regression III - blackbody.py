import pandas as pd
import numpy as np
from pysr import PySRRegressor
import os

# Constants
PLANCK_CONSTANT = 6.62607015e-34  # Planck's constant (J·s)
BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant (J/K)
SPEED_OF_LIGHT = 3.0e8  # Speed of light (m/s)

# Check if the dataset exists, else create one
if not os.path.exists("blackbody_radiation_data.csv"):
    print("Dataset not found! Generating synthetic dataset...")

    # Synthetic data generation
    np.random.seed(42)
    temperatures = np.linspace(300, 10000, 1000)  # Temperature range in Kelvin
    wavelengths = np.linspace(1e-7, 3e-6, 1000)  # Wavelengths in meters

    # Calculate blackbody intensity using Planck's law
    def plancks_law(T, lambda_):
        return (2 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2 / lambda_**5) / \
               (np.exp((PLANCK_CONSTANT * SPEED_OF_LIGHT) / (lambda_ * BOLTZMANN_CONSTANT * T)) - 1)

    # Generate intensity for each temperature and wavelength
    data = []
    for T in temperatures:
        for lambda_ in wavelengths:
            intensity = plancks_law(T, lambda_)
            data.append([T, lambda_, intensity])

    # Create a DataFrame
    blackbody_data = pd.DataFrame(data, columns=["Temperature", "Wavelength", "Intensity"])

    # Save the synthetic dataset
    blackbody_data.to_csv("blackbody_radiation_data.csv", index=False)
    print("Synthetic dataset created as 'blackbody_radiation_data.csv'.")

# Load the dataset
blackbody_data = pd.read_csv("blackbody_radiation_data.csv")

# Feature engineering for layered universality
blackbody_data['Scaled_Intensity'] = blackbody_data['Intensity'] / blackbody_data['Intensity'].max()
blackbody_data['Log_Temperature'] = np.log(blackbody_data['Temperature'])
blackbody_data['Log_Wavelength'] = np.log(blackbody_data['Wavelength'])

# Variables: Temperature (T), Wavelength (λ), and Scaled Intensity
X = blackbody_data[["Temperature", "Wavelength", "Log_Temperature", "Log_Wavelength"]].values
y = blackbody_data["Scaled_Intensity"].values

model = PySRRegressor(
    model_selection="best",
    niterations=10000,  # Reduced for faster execution
    populations=20,
    equation_file="blackbody_symbolic_equations.csv",
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "cos", "sin", "exp", "log", "sqrt", "abs",
        "square(x) = x^2", "inv(x) = 1/x"
    ],
    extra_sympy_mappings={"square": lambda x: x**2, "inv": lambda x: 1/x},
    constraints={"^": (-2, 2)},  # Constrain power complexity
    deterministic=True,  # Ensure consistent results
    batching=True,  # Enable batching for large datasets
    batch_size=1000,  # Adjust based on dataset size
    procs=0,  # Single process for determinism
    multithreading=False,  # Disable multithreading for determinism
    verbosity=1,
    random_state=42,  # Set seed for reproducibility
)

# Fit the model
print("Fitting the symbolic regression model...")
model.fit(X, y)


# Print the best equations
print("Best equations:")
print(model)

# Save equations for further analysis
equations = pd.DataFrame(model.equations_)
equations.to_csv("refined_blackbody_equations.csv", index=False)

# Analyze universality across domains
# Use quantum constants for scaling
blackbody_data['Predicted_Intensity'] = model.predict(X)
blackbody_data['Quantum_Scale'] = PLANCK_CONSTANT * SPEED_OF_LIGHT / blackbody_data['Wavelength']
blackbody_data['Classical_Scale'] = BOLTZMANN_CONSTANT * blackbody_data['Temperature']

# Compare predictions with observed intensity
print("Preview of predictions vs actual intensity:")
print(blackbody_data[["Temperature", "Wavelength", "Intensity", "Predicted_Intensity"]].head(10))

# Save enhanced dataset
blackbody_data.to_csv("Enhanced_Blackbody_Radiation_with_Predictions.csv", index=False)

# Plot results for visual analysis
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.loglog(blackbody_data['Wavelength'], blackbody_data['Intensity'], label="Observed Intensity")
plt.loglog(blackbody_data['Wavelength'], blackbody_data['Predicted_Intensity'], label="Predicted Intensity")
plt.xlabel("Wavelength (m)")
plt.ylabel("Intensity (W/m^2/sr)")
plt.title("Observed vs Predicted Intensity (Blackbody Radiation)")
plt.legend()
plt.grid()
plt.savefig("Blackbody_Regression_Comparison.png")
plt.show()


#inconclusive may need restrainig parameters, and new dataset