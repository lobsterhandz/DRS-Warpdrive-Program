from generate_data import generate_theoretical_spectra, simulate_cmb_map, plot_cmb_map
from preprocess import preprocess_data
from symbolic_model import symbolic_regression
from overlay_models import overlay_models
import numpy as np

# Generate theoretical spectra and simulate a CMB map
lmax = 2000
cl = generate_theoretical_spectra(lmax=lmax)
cmb_map = simulate_cmb_map(cl)
plot_cmb_map(cmb_map, title="Simulated CMB Map")

# Preprocess the data
noisy_map, cl_preprocessed = preprocess_data(cmb_map)

# Prepare features and target for symbolic regression
temperature = np.linspace(2.7, 300, len(cl_preprocessed))  # Example temperatures
X = np.column_stack([temperature, cl_preprocessed])
y = cl_preprocessed  # Target is the preprocessed spectrum

# Run symbolic regression
model = symbolic_regression(X, y)

# Predictions and overlay
predicted = [model.predict(X)]
overlay_models(temperature, y, predicted, labels=["Symbolic Model"])
