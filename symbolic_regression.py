from pysr import PySRRegressor
import pandas as pd

# Load the dataset
data = pd.read_csv('phase_transition_data.csv')  # Update with your file path

# Prepare data for symbolic regression
X = data[['Lambda', 'Dim_Factor']].values
y_entropy = data['Entropy'].values
y_variance = data['Variance'].values

# Symbolic regression for Entropy
model_entropy = PySRRegressor(
    niterations=100,
    unary_operators=["sin", "cos", "exp", "log"],
    binary_operators=["+", "-", "*", "/"],
    verbosity=1,
)
model_entropy.fit(X, y_entropy)

# Symbolic regression for Variance
model_variance = PySRRegressor(
    niterations=100,
    unary_operators=["sin", "cos", "exp", "log"],
    binary_operators=["+", "-", "*", "/"],
    verbosity=1,
)
model_variance.fit(X, y_variance)

# Display equations
print("Entropy Equation:", model_entropy.get_best())
print("Variance Equation:", model_variance.get_best())
