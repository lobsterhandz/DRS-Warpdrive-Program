import pandas as pd
import numpy as np
from pysr import PySRRegressor
import os

# Check if the dataset exists, else create one
if not os.path.exists("thermal_systems_data.csv"):
    print("Dataset not found! Generating synthetic dataset...")

    # Synthetic data generation
    np.random.seed(42)
    num_samples = 1000

    # Variables: S = entropy, T = temperature, E = energy, etc.
    S = np.random.uniform(1, 100, num_samples)  # Entropy
    T = np.random.uniform(1, 300, num_samples)  # Temperature in Kelvin
    E = S * T  # Approximate energy as a product of entropy and temperature
    k_B = 1.380649e-23 * np.ones(num_samples)  # Boltzmann constant (J/K)
    h = 6.62607015e-34 * np.ones(num_samples)  # Planck constant (J·s)

    # Observed Variance (target variable): Example with added noise
    observed_variance = (k_B * T * np.log(S)) + np.random.normal(0, 0.01, num_samples)

    # Create synthetic dataset
    synthetic_data = pd.DataFrame({
        "S": S,
        "T": T,
        "E": E,
        "k_B": k_B,
        "h": h,
        "Observed_Variance": observed_variance,
        "System_Type": np.random.choice(["Quantum", "Classical"], num_samples),
    })

    # Save to CSV
    synthetic_data.to_csv("thermal_systems_data.csv", index=False)
    print("Synthetic dataset created as 'thermal_systems_data.csv'.")

# Load the dataset
data = pd.read_csv("thermal_systems_data.csv")

# Variables: entropy (S), temperature (T), energy (E), etc.
X = data[["S", "T", "E", "k_B", "h"]].values  # Features
y = data["Observed_Variance"].values          # Target

model = PySRRegressor(
    model_selection="best",
    niterations=1000,
    populations=20,
    equation_file="symbolic_regression_thermal_equations.csv",
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "cos", "sin", "exp", "log", 
        "sqrt", "abs", 
        "square(x) = x^2"
    ],
    extra_sympy_mappings={"square": lambda x: x**2},  # Custom operator mapping
    constraints={"^": (-1, 1)},  # Limit power complexity
    deterministic=True,  # Ensure consistent results
    verbosity=1,
    random_state=42,  # Ensures reproducibility
    procs=0,          # Single-threaded for determinism
    multithreading=False,  # Disable multithreading
)

# Fit the model to the data
print("Fitting the symbolic regression model...")
model.fit(X, y)

# Print and save the results
print("Best equations:")
print(model)

# Save equations for further analysis
equations = pd.DataFrame(model.equations_)
equations.to_csv("refined_thermal_equations.csv", index=False)

# Evaluate convergence between quantum and classical systems
quantum_data = data[data["System_Type"] == "Quantum"]
classical_data = data[data["System_Type"] == "Classical"]

quantum_preds = model.predict(quantum_data[["S", "T", "E", "k_B", "h"]].values)
classical_preds = model.predict(classical_data[["S", "T", "E", "k_B", "h"]].values)

# Compare predictions for universality
print("Quantum predictions vs Classical predictions:")
print("Quantum:", quantum_preds[:5])
print("Classical:", classical_preds[:5])

# Save results for further analysis
np.savetxt("quantum_predictions.csv", quantum_preds, delimiter=",")
np.savetxt("classical_predictions.csv", classical_preds, delimiter=",")
