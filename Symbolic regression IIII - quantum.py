import pandas as pd
import numpy as np
from pysr import PySRRegressor

# Synthetic dataset for demonstration
np.random.seed(42)
num_samples = 1000

# Quantum dataset variables
S_quantum = np.random.uniform(1, 100, num_samples)  # Entropy
T_quantum = np.random.uniform(1, 300, num_samples)  # Temperature in Kelvin
E_quantum = S_quantum * T_quantum                   # Energy approximation
observed_variance_quantum = 1.380649e-23 * T_quantum * np.log(S_quantum) + np.random.normal(0, 0.01, num_samples)

# Create a quantum system dataset
quantum_data = pd.DataFrame({
    "S": S_quantum,
    "T": T_quantum,
    "E": E_quantum,
    "Observed_Variance": observed_variance_quantum
})

# Updated PySR model with deterministic settings
model_quantum = PySRRegressor(
    model_selection="best",
    niterations=1000,
    populations=20,
    equation_file="symbolic_regression_equations.csv",
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["cos", "sin", "exp", "log", "sqrt", "abs", "square(x) = x^2"],
    extra_sympy_mappings={"square": lambda x: x**2},
    constraints={"^": (-1, 1)},  # Limit power complexity
    deterministic=True,
    random_state=42,  # Set seed for reproducibility
    procs=0,  # Single-threaded processing
    multithreading=False,  # Disable multithreading
    verbosity=1,
)


# Prepare data for regression
X_quantum = quantum_data[["S", "T", "E"]].values
y_quantum = quantum_data["Observed_Variance"].values

# Fit the model
try:
    print("Fitting symbolic regression for quantum data...")
    model_quantum.fit(X_quantum, y_quantum)
    
    # Save results in memory
    quantum_equations = pd.DataFrame(model_quantum.equations_)
    quantum_results = {
        "equations": quantum_equations,
        "predictions": model_quantum.predict(X_quantum)
    }
    
    # Display the first few equations and predictions
    print("Top equations:")
    print(quantum_equations.head())
    print("Sample predictions:")
    print(quantum_results["predictions"][:5])
    
except Exception as e:
    print("An error occurred during symbolic regression:", e)
