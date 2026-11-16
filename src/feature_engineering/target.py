from pandas import Series
from src.config import TARGET_MAP

def encode_target (target : Series) -> Series:
    return target.map(TARGET_MAP)


def decode_target (target : Series) -> Series:
    inverse_map = {value : key for key, value in TARGET_MAP.items()}

    return target.map(inverse_map)
