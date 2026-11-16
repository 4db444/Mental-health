from joblib import dump
from pandas import DataFrame, Series, crosstab
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    MODEL_REPORT_PATHS,
    TARGET_CATEGORIES,
)
from ..data import save_dataset
from ..feature_engineering import decode_target
from ..visualization import plot_confusion_matrix

def _classification_metrics (model, x : DataFrame, y : Series) -> dict[str : float]:
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)

    return {
        "accuracy" : round(accuracy_score(y, predictions), 4),
        "precision" : round(precision_score(y, predictions, average="weighted", zero_division=0), 4),
        "recall" : round(recall_score(y, predictions, average="weighted", zero_division=0), 4),
        "f1" : round(f1_score(y, predictions, average="weighted", zero_division=0), 4),
        "roc auc" : round(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"), 4),
    }


def evaluate_models (models : list[dict], x : DataFrame, y : Series, filename : str = "baseline metrics.csv") -> DataFrame:
    report = {
        result["name"] : _classification_metrics(result["model"], x, y)
        for result in models
    }

    report_df = DataFrame(report).T

    print("Models Metrics.")
    print(report_df.to_string())

    save_dataset(report_df, MODEL_REPORT_PATHS, filename, True)

    return report_df


def optimization_report (results : list[dict], filename : str = "optimization report.csv") -> DataFrame:
    report = {
        result["name"] : {
            "best cv f1" : round(result["best_score"], 4),
            "best params" : str(result["best_params"]),
        }
        for result in results
    }

    report_df = DataFrame(report).T

    print("Optimization Report.")
    print(report_df.to_string())

    save_dataset(report_df, MODEL_REPORT_PATHS, filename, True)

    return report_df


def select_best (results : list[dict]) -> dict:
    best_result = max(results, key=lambda result : result["best_score"])

    print(f"Best model: {best_result['name']}")
    print(f"Best cv f1: {best_result['best_score']:.4f}")
    print(f"Best params: {best_result['best_params']}")

    return best_result

def evaluate_final (model, x_test : DataFrame, y_test : Series, model_name : str) -> dict:
    predictions = model.predict(x_test)
    stats = _classification_metrics(model, x_test, y_test)

    report_df = DataFrame(
        classification_report(
            y_test,
            predictions,
            target_names=TARGET_CATEGORIES,
            output_dict=True,
        )
    ).T.round(4)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    matrix_df = DataFrame(matrix, index=TARGET_CATEGORIES, columns=TARGET_CATEGORIES)

    print("Final Test Metrics.")
    print(DataFrame(stats, index=[model_name]).to_string())

    print("Final Classification Report.")
    print(report_df.to_string())

    save_dataset(DataFrame(stats, index=[model_name]), MODEL_REPORT_PATHS, "final metrics.csv", True)
    save_dataset(report_df, MODEL_REPORT_PATHS, "final classification report.csv", True)
    save_dataset(matrix_df, MODEL_REPORT_PATHS, "final confusion matrix.csv", True)

    plot_confusion_matrix(matrix, model_name)

    return {
        "name" : model_name,
        "stats" : stats,
        "confusion_matrix" : matrix,
        "report" : report_df,
    }


def analyze_errors (model, x_test : DataFrame, y_test : Series, model_name : str) -> DataFrame:
    predictions = model.predict(x_test)

    errors_df = x_test.copy()
    errors_df["actual"] = decode_target(y_test)
    errors_df["predicted"] = decode_target(Series(predictions, index=y_test.index))
    errors_df = errors_df[errors_df["actual"] != errors_df["predicted"]]

    print(f"Misclassified rows: {len(errors_df)} / {len(y_test)}")

    if len(errors_df):
        print(crosstab(errors_df["actual"], errors_df["predicted"]).to_string())

    save_dataset(errors_df, MODEL_REPORT_PATHS, f"errors - {model_name}.csv", True)

    return errors_df


def save_model (model, model_name : str) -> None:
    artifact = {
        "model" : model,
        "best_model_name" : model_name,
        "feature_columns" : FEATURE_COLUMNS,
        "target_categories" : TARGET_CATEGORIES,
    }

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, FINAL_MODEL_PATH)

    print(f"Model saved to: {FINAL_MODEL_PATH}")
