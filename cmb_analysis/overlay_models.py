import matplotlib.pyplot as plt
import numpy as np

def overlay_models(temperature, observed, predicted, labels):
    """Overlay observed and predicted data."""
    plt.figure(figsize=(10, 6))
    for i, pred in enumerate(predicted):
        plt.plot(temperature, pred, label=labels[i], linestyle='--')
    plt.plot(temperature, observed, label="Observed", linestyle="-", color="black")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Variance")
    plt.title("Observed vs Predicted Models")
    plt.legend()
    plt.show()
