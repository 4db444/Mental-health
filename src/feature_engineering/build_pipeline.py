from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from src.config import (
    BINARY_FEATURES,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    RANDOM_SEED,
)

def build_preprocessor () -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=list(ORDINAL_FEATURES.values()),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["No", "Yes"] for _ in BINARY_FEATURES],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("ordinal", ordinal_pipeline, list(ORDINAL_FEATURES)),
        ("binary", binary_pipeline, BINARY_FEATURES),
        ("nominal", nominal_pipeline, NOMINAL_FEATURES),
    ])


def build_pipeline (classifier, use_smote : bool = False) -> Pipeline:
    steps = [("preprocessor", build_preprocessor())]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))

    steps.append(("classifier", classifier))

    return Pipeline(steps)
