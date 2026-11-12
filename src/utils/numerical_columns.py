from pandas import DataFrame

def numerical_cols (df : DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns