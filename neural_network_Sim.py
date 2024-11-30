import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load the data and trained neural network model
file_path = 'parameter_sweep_results.csv'  # Update this with your file path
data = pd.read_csv(file_path)
model = load_model('trained_model.h5')  # Ensure you save your trained model

# Features and target
X = data[['lambda_res', 'gamma', 'higher_dim_factor', 'A_final', 'B_final']]
y = data['lyapunov_exp']

# Standardize the input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to analyze feature sensitivity
def feature_sensitivity_analysis(model, X_scaled, y, feature_names):
    predictions = model.predict(X_scaled).flatten()
    mse_baseline = mean_squared_error(y, predictions)

    # Permutation importance
    results = permutation_importance(
        model,
        X_scaled,
        y,
        scoring='neg_mean_squared_error',
        n_repeats=10,
        random_state=42
    )

    # Visualize feature importances
    importances = results.importances_mean
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel("Mean Importance")
    plt.title("Feature Sensitivity Analysis")
    plt.show()

    return importances

# Perform sensitivity analysis
feature_names = ['lambda_res', 'gamma', 'higher_dim_factor', 'A_final', 'B_final']
sensitivity_importances = feature_sensitivity_analysis(model, X_scaled, y, feature_names)

# Function to generate phase-space diagrams
def plot_phase_space(data, x_param, y_param, target_param):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[x_param], data[y_param], c=data[target_param], cmap='viridis', alpha=0.75)
    plt.colorbar(scatter, label=target_param)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"Phase Space: {x_param} vs {y_param} ({target_param} as color)")
    plt.show()

# Generate phase-space diagrams
plot_phase_space(data, 'lambda_res', 'gamma', 'lyapunov_exp')
plot_phase_space(data, 'higher_dim_factor', 'A_final', 'lyapunov_exp')

# Function to extract and visualize learned weights
def visualize_neural_weights(model):
    for i, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        print(f"Layer {i+1} - Weights shape: {weights.shape}, Biases shape: {biases.shape}")

        plt.figure(figsize=(10, 5))
        plt.title(f"Weights for Layer {i+1}")
        plt.imshow(weights, cmap='viridis', aspect='auto')
        plt.colorbar(label='Weight Magnitude')
        plt.xlabel("Neuron")
        plt.ylabel("Input Feature" if i == 0 else "Previous Layer Neuron")
        plt.show()

# Visualize the weights of the neural network
visualize_neural_weights(model)

# Function to validate predictions using DRS-inspired thresholds
def validate_predictions_with_drs(data, model, threshold=37.0):
    predictions = model.predict(X_scaled).flatten()
    data['predicted_lyapunov'] = predictions

    # Filter predictions based on DRS thresholds
    stable_predictions = data[data['predicted_lyapunov'] < threshold]
    chaotic_predictions = data[data['predicted_lyapunov'] >= threshold]

    print(f"Stable Predictions: {len(stable_predictions)}")
    print(f"Chaotic Predictions: {len(chaotic_predictions)}")

    plt.figure(figsize=(8, 6))
    plt.hist(predictions, bins=20, color='skyblue', alpha=0.7, label='Predicted Lyapunov')
    plt.axvline(threshold, color='red', linestyle='--', label='DRS Stability Threshold')
    plt.xlabel("Predicted Lyapunov Exponent")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Validation of Predictions Against DRS Threshold")
    plt.show()

# Validate predictions
validate_predictions_with_drs(data, model)
