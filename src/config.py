from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# contants
HOURS_IN_WEEK = 24 * 7
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
IMBALANCE_THRESHOLD = 1

# data paths
RAW_DATA_FILE = "student_mental_health.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_PATH / TRAIN_DATA_FILE
TEST_DATA_PATH = PROCESSED_DATA_PATH / TEST_DATA_FILE

# report paths
EDA_REPORT_PATHS = PROJECT_ROOT / "reports" / "eda"
MODEL_REPORT_PATHS = PROJECT_ROOT / "reports" / "models"
FIGURES_PATH = PROJECT_ROOT / "figures"
MODELS_PATH = PROJECT_ROOT / "models"
OPTIMIZED_MODEL_PATH = MODELS_PATH / "optimized_model.joblib"
FINAL_MODEL_PATH = MODELS_PATH / "final_model.joblib"

# columns
ID_COL = "Student_ID"
TARGET_COL = "Mental_Health_Risk"
TARGET_CATEGORIES = ["Low", "Medium", "High"]
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}

NUMERICAL_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

ORDINAL_FEATURES = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}

BINARY_FEATURES = [
    "Part_Time_Job",
    "Previous_Mental_Health_History",
]

NOMINAL_FEATURES = ["Gender"]
NOMINAL_CATEGORIES = {"Gender": ["Female", "Male", "Non-binary"]}

CATEGORICAL_FEATURES = (
    list(ORDINAL_FEATURES) + BINARY_FEATURES + NOMINAL_FEATURES
)

CATEGORY_OPTIONS = {
    **ORDINAL_FEATURES,
    **{col : ["No", "Yes"] for col in BINARY_FEATURES},
    **NOMINAL_CATEGORIES,
}

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
