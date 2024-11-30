import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor

# Step 1: Generate Synthetic Data for Gravitational System
np.random.seed(42)
num_samples = 1000
time = np.linspace(0, 10, num_samples)  # Time in seconds
amplitude = np.exp(-0.1 * time) * np.cos(2 * np.pi * time)  # Damped oscillation
frequency = 2 * np.pi / time[1:]  # Simplified frequency scaling
gravitational_constant = 6.67430e-11 * np.ones(num_samples)  # G

# Observed Variance (Synthetic Target Variable)
observed_variance = (amplitude * gravitational_constant) + np.random.normal(0, 0.001, num_samples)

# Step 2: Create DataFrame
gravitational_data = pd.DataFrame({
    "Time": time,
    "Amplitude": amplitude,
    "Frequency": np.append(frequency, 0),  # Pad frequency for consistency
    "G": gravitational_constant,
    "Observed_Variance": observed_variance
})

# Step 3: Symbolic Regression
X = gravitational_data[["Time", "Amplitude", "Frequency", "G"]].values
y = gravitational_data["Observed_Variance"].values

model = PySRRegressor(
    model_selection="best",
    niterations=1000,
    populations=20,
    equation_file="gravitational_equations.csv",
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log"],
    constraints={"^": (-1, 1)},  # Restrict power complexity
    deterministic=True,
    random_state=42,  # Set a fixed seed
    procs=0,  # Disable multiprocessing
    multithreading=False,  # Disable multithreading
    verbosity=1  # Increase verbosity for debugging
)


print("Fitting symbolic regression model...")
model.fit(X, y)

# Step 4: Analyze and Visualize
print("Best equations:")
print(model)

# Plot predictions
predictions = model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(time, y, label="Observed Variance", linestyle="--", color="blue")
plt.plot(time, predictions, label="Symbolic Predictions", linestyle="-", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Variance")
plt.legend()
plt.title("Gravitational System: Observed vs Predicted")
plt.show()

# Save the results
gravitational_data["Predicted_Variance"] = predictions
gravitational_data.to_csv("Gravitational_System_Results.csv", index=False)
