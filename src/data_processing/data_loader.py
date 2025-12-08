import pandas as pd
from pathlib import Path

def load_data(filepath: str, sep=",", low_memory=False):
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return pd.read_csv(file_path, sep=sep, low_memory=low_memory)
