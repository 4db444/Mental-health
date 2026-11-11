from pathlib import Path
import pandas as pd
from pandas import DataFrame

def laod_dataset (path : Path, filename : str) -> DataFrame:
    return pd.read_csv(path / filename)