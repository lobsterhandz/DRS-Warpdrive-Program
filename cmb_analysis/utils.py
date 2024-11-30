import pandas as pd

def save_to_csv(data, filename):
    """Save data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_csv(filename):
    """Load data from a CSV file."""
    return pd.read_csv(filename)
