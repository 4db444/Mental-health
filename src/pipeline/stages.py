from joblib import dump, load
from pandas import DataFrame, Series, read_csv
from src.config import (
    IMBALANCE_THRESHOLD,
    OPTIMIZED_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_FILE,
    RAW_DATA_PATH,
    TEST_DATA_FILE,
    TEST_DATA_PATH,
    TRAIN_DATA_FILE,
    TRAIN_DATA_PATH,
    TARGET_COL,
)
from ..data import (
    categorical_target_summary,
    correlation,
    descriptive_statistics,
    duplicate_values,
    inconsistencies,
    laod_dataset,
    missing_values,
    numerical_target_summary,
    outliers,
    save_dataset,
    split_dataset,
    target_distribution,
)
from ..feature_engineering import encode_target
from ..train import (
    analyze_errors,
    evaluate_final,
    evaluate_models,
    optimization_report,
    optimize_models,
    save_model,
    select_best,
    train_models,
)
from ..visualization import plot_correlation, plot_model_comparison

def _load_raw_data () -> DataFrame:
    return laod_dataset(RAW_DATA_PATH, RAW_DATA_FILE)


def _load_processed_data () -> tuple[DataFrame, DataFrame]:
    train_df = read_csv(TRAIN_DATA_PATH)
    test_df = read_csv(TEST_DATA_PATH)

    return train_df, test_df


def _split_features_target (df : DataFrame) -> tuple[DataFrame, Series]:
    x = df.drop(columns=[TARGET_COL])
    y = encode_target(df[TARGET_COL])

    return x, y


def _needs_smote (target : Series) -> bool:
    counts = target.value_counts()

    return counts.max() / counts.min() > IMBALANCE_THRESHOLD


def prepare_data () -> None:
    df = _load_raw_data()

    missing_values(df)
    duplicate_values(df)
    inconsistencies(df)
    outliers(df)

    descriptive_statistics(df)
    target_distribution(df)
    numerical_target_summary(df)
    categorical_target_summary(df)
    plot_correlation(correlation(df))

    print("Data preparation finished.")


def preprocess_data () -> None:
    df = _load_raw_data()
    x_train, x_test, y_train, y_test = split_dataset(df)

    train_df = x_train.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = x_test.copy()
    test_df[TARGET_COL] = y_test.values

    save_dataset(train_df, PROCESSED_DATA_PATH, TRAIN_DATA_FILE)
    save_dataset(test_df, PROCESSED_DATA_PATH, TEST_DATA_FILE)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


def train_stage () -> None:
    train_df, _ = _load_processed_data()
    x_train, y_train = _split_features_target(train_df)
    use_smote = _needs_smote(y_train)

    models = train_models(x_train, y_train, use_smote)
    report = evaluate_models(models, x_train, y_train, "baseline metrics.csv")
    plot_model_comparison(report, filename="baseline model comparison.png")

    print(f"SMOTE enabled: {use_smote}")


def optimize_stage () -> None:
    train_df, _ = _load_processed_data()
    x_train, y_train = _split_features_target(train_df)
    use_smote = _needs_smote(y_train)

    results = optimize_models(x_train, y_train, use_smote)
    report = optimization_report(results)
    plot_model_comparison(report, metric="best cv f1", filename="optimized model comparison.png")

    best = select_best(results)
    dump(best, OPTIMIZED_MODEL_PATH)

    print(f"Optimized model saved to: {OPTIMIZED_MODEL_PATH}")


def evaluate_stage () -> None:
    _, test_df = _load_processed_data()
    x_test, y_test = _split_features_target(test_df)

    best = load(OPTIMIZED_MODEL_PATH)

    evaluate_final(best["model"], x_test, y_test, best["name"])
    analyze_errors(best["model"], x_test, y_test, best["name"])
    save_model(best["model"], best["name"])
