from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from src.config import ID_COL, RANDOM_SEED, TARGET_COL, TEST_SIZE

def split_dataset (df : DataFrame, test_size : float = TEST_SIZE, random_state : int = RANDOM_SEED) -> tuple[DataFrame, DataFrame, Series, Series]:
    x = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
