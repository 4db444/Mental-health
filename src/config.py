from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# report paths
EDA_REPORT_PATHS = PROJECT_ROOT / "reports" / "eda"
