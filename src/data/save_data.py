from pathlib import Path
from pandas import DataFrame

def save_dataset(df : DataFrame, path : Path, filename : str, index : bool = False) -> None : 
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / filename, index=index)
