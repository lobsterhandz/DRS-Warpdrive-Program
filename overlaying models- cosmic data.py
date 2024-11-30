import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor

# Step 1: Load cosmic dataset (replace with actual dataset paths)
# Example: Cosmic Microwave Background (CMB) data
cosmic_data = pd.read_csv("cosmic_phenomena_data.csv")  # Ensure dataset availability

# Step 2: Extract variables (adjust according to your dataset structure)
temperature = cosmic_data["Temperature"].values  # Example: temperature in Kelvin
intensity = cosmic_data["Intensity"].values  # Example: observed intensity
distance = cosmic_data["Distance"].values  # Example: distance in parsecs
time = cosmic_data["Time"].values  # Example: time in seconds (if applicable)

# Features for symbolic regression models
X_cosmic = np.column_stack([temperature, intensity, distance, time])

# Step 3: Load symbolic regression models (adjust equations as needed)
symbolic_equations = [
    lambda x: x[1] * 0.00012399,  # Simplest linear equation
    lambda x: -9.051e-5 * np.cos(x[0]),
    lambda x: -0.00010521 * np.sin(np.cos(x[0])),
    lambda x: (np.cos(x[0] * -1.1152) - x[1]) * -0.00012227,
    lambda x: -0.00013838 * np.cos(-0.68861 - (x[0] - np.cos(x[1] / x[2])))
]

# Step 4: Predict using symbolic models
predictions = {}
for i, equation in enumerate(symbolic_equations):
    predictions[f"Equation {i+1}"] = np.apply_along_axis(equation, 1, X_cosmic)

# Step 5: Plot observed vs predicted data
plt.figure(figsize=(10, 6))

# Observed data
plt.plot(temperature, intensity, 'k-', label="Observed Intensity (Cosmic Data)")

# Predicted data from symbolic models
for i, (label, pred) in enumerate(predictions.items()):
    plt.plot(temperature, pred, linestyle="--", label=f"Prediction: {label}")

plt.xlabel("Temperature (K)")
plt.ylabel("Intensity")
plt.title("Cosmic Phenomena: Observed vs Symbolic Model Predictions")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Save results for analysis
cosmic_predictions = pd.DataFrame(predictions)
cosmic_predictions["Observed_Intensity"] = intensity
cosmic_predictions.to_csv("Cosmic_Phenomena_Symbolic_Overlay.csv", index=False)
print("Overlay results saved as 'Cosmic_Phenomena_Symbolic_Overlay.csv'")
