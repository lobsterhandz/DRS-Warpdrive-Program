import numpy as np
import pandas as pd

# Define synthetic data parameters
np.random.seed(42)  # For reproducibility
x_values = np.linspace(0.1, 10, 100)  # Range of x values

# Generate synthetic entropy and variance data
entropy_values = 7 * np.log(x_values + 1) + np.random.normal(0, 0.1, size=len(x_values))
variance_values = 5 * np.sin(2 * np.pi * x_values / 10) + np.random.normal(0, 0.1, size=len(x_values))

# Create a DataFrame
data = {
    "x": x_values,
    "entropy": entropy_values,
    "variance": variance_values
}
df = pd.DataFrame(data)

# Save to CSV
csv_file_name = "quantum_thermal_data.csv"
df.to_csv(csv_file_name, index=False)

print(f"Data successfully written to {csv_file_name}")
