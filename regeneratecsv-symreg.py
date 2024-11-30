import pandas as pd
import numpy as np

# Simulated data
lambda_values = np.linspace(0.1, 1.0, 10)
dim_factors = np.linspace(0.1, 1.0, 10)
entropy_values = np.random.rand(10, 10) * 5 + 4
variance_values = np.random.rand(10, 10) * 2

data = pd.DataFrame({
    'Lambda': np.repeat(lambda_values, len(dim_factors)),
    'Dim_Factor': np.tile(dim_factors, len(lambda_values)),
    'Entropy': entropy_values.flatten(),
    'Variance': variance_values.flatten(),
})

# Save to CSV
data.to_csv('phase_transition_data.csv', index=False)
