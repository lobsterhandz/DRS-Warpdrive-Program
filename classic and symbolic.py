# Overlay quantum and classical results for convergence analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load symbolic predictions for quantum and classical systems
quantum_preds = np.loadtxt("quantum_predictions.csv", delimiter=",")
classical_preds = np.loadtxt("classical_predictions.csv", delimiter=",")

# Load the original dataset to get independent variables (Temperature, Entropy, etc.)
data = pd.read_csv("thermal_systems_data.csv")

# Extract relevant variables
temperature = data["T"]
entropy = data["S"]
energy = data["E"]

# Filter temperature to match the length of classical_preds
# Adjust slicing or filtering logic to ensure proper alignment
matching_temperature_classical = temperature[:len(classical_preds)]  # Adjust based on actual data filtering logic

# Filter temperature to match quantum_preds
matching_temperature_quantum = temperature[:len(quantum_preds)]

# Plot Quantum Predictions
plt.plot(matching_temperature_quantum, quantum_preds, label="Quantum Predictions", linestyle="--", color="blue")

# Plot Classical Predictions
plt.plot(matching_temperature_classical, classical_preds, label="Classical Predictions", linestyle="-", color="red")

# Add labels and legend
plt.xlabel("Temperature (K)")
plt.ylabel("Predictions")
plt.legend()
plt.title("Quantum vs Classical Predictions")
plt.grid()
plt.show()


# Save the combined plot data for further analysis
overlay_data = pd.DataFrame({
    "Temperature": temperature,
    "Quantum Predictions": quantum_preds,
    "Classical Predictions": classical_preds,
    "Observed Variance": data["Observed_Variance"]
})
overlay_data.to_csv("Quantum_Classical_Overlay.csv", index=False)
print(f"Length of temperature: {len(temperature)}")
print(f"Length of quantum_preds: {len(quantum_preds)}")
print(f"Length of classical_preds: {len(classical_preds)}")
