from pysr import PySRRegressor
import numpy as np
import pandas as pd

def symbolic_regression(X, y, equations_file="symbolic_equations.csv"):
    """Run symbolic regression on given features and target."""
    model = PySRRegressor(
        model_selection="best",
        niterations=1000,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["cos", "sin", "exp", "log", "sqrt"],
        deterministic=True,
        random_state=42,
        equation_file=equations_file,
        verbosity=1,
    )
    model.fit(X, y)
    return model
