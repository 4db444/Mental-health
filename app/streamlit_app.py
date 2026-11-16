from joblib import load
from pandas import DataFrame, Series, read_csv
from streamlit import (
    bar_chart,
    columns,
    dataframe,
    error,
    form,
    form_submit_button,
    header,
    image,
    metric,
    number_input,
    selectbox,
    set_page_config,
    sidebar,
    success,
    tabs,
    title,
    warning,
)
from src import decode_target
from src.config import (
    CATEGORY_OPTIONS,
    FIGURES_PATH,
    FINAL_MODEL_PATH,
    MODEL_REPORT_PATHS,
    NUMERICAL_FEATURES,
    TARGET_CATEGORIES,
)

NUMERICAL_INPUTS = {
    "Age" : (17, 29, 21),
    "Year_of_Study" : (1, 5, 2),
    "Sleep_Hours" : (3.3, 10.0, 6.7),
    "Study_Hours_Per_Week" : (2.4, 38.3, 19.8),
    "GPA" : (1.5, 3.66, 2.56),
    "Social_Media_Hours" : (2.1, 10.0, 6.0),
    "Screen_Time_Hours" : (3.3, 14.0, 8.5),
    "Financial_Stress" : (1.4, 10.0, 6.1),
    "Family_Support" : (1.7, 9.9, 6.7),
    "Attendance_Rate" : (54.4, 100.0, 85.1),
}

set_page_config(page_title="Student Mental Health Risk", layout="wide")
title("Student Mental Health Risk")

artifact = load(FINAL_MODEL_PATH) if FINAL_MODEL_PATH.exists() else None

sidebar.header("About")
sidebar.write(
    "This app reports the performance of the trained classifier and "
    "estimates the mental health risk level of a new student."
)
sidebar.write("The model predicts a risk band, it is not a medical diagnosis.")

performance_tab, predict_tab = tabs(["Model Performance", "Predict Risk"])

with performance_tab:
    metrics_path = MODEL_REPORT_PATHS / "final metrics.csv"

    if not metrics_path.exists():
        warning("No metrics found. Run `python -m src.pipeline all` first.")
    else:
        metrics_df = read_csv(metrics_path, index_col=0)
        model_name = metrics_df.index[0]

        header(f"Final Test Metrics - {model_name}")

        metric_columns = columns(4)
        metric_columns[0].metric("Accuracy", f"{metrics_df.loc[model_name, 'accuracy']:.4f}")
        metric_columns[1].metric("Precision", f"{metrics_df.loc[model_name, 'precision']:.4f}")
        metric_columns[2].metric("Recall", f"{metrics_df.loc[model_name, 'recall']:.4f}")
        metric_columns[3].metric("F1-score", f"{metrics_df.loc[model_name, 'f1']:.4f}")

        metric_columns = columns(2)
        metric_columns[0].metric("ROC-AUC", f"{metrics_df.loc[model_name, 'roc auc']:.4f}")
        metric_columns[1].metric("Model", model_name)

        dataframe(metrics_df)

        header("Confusion Matrix")
        matrix_path = FIGURES_PATH / f"confusion matrix - {model_name}.png"

        if matrix_path.exists():
            image(str(matrix_path), width="stretch")

        matrix_data_path = MODEL_REPORT_PATHS / "final confusion matrix.csv"

        if matrix_data_path.exists():
            dataframe(read_csv(matrix_data_path, index_col=0))

        header("Classification Report")
        report_path = MODEL_REPORT_PATHS / "final classification report.csv"

        if report_path.exists():
            dataframe(read_csv(report_path, index_col=0))

        header("Hyperparameter Optimization")
        optimization_path = MODEL_REPORT_PATHS / "optimization report.csv"

        if optimization_path.exists():
            dataframe(read_csv(optimization_path, index_col=0))

        header("EDA & Model Comparison")
        figure_columns = columns(2)

        with figure_columns[0]:
            correlation_path = FIGURES_PATH / "correlation heatmap.png"

            if correlation_path.exists():
                image(str(correlation_path), caption="Feature correlation", width="stretch")

        with figure_columns[1]:
            comparison_path = FIGURES_PATH / "optimized model comparison.png"

            if comparison_path.exists():
                image(str(comparison_path), caption="Model comparison", width="stretch")

with predict_tab:
    header("Test the model on a new student")

    if artifact is None:
        warning("No saved model found. Run `python -m src.pipeline all` first.")
    else:
        with form("prediction_form"):
            input_columns = columns(2)
            row = {}

            for index, col in enumerate(NUMERICAL_FEATURES):
                minimum, maximum, default = NUMERICAL_INPUTS[col]

                with input_columns[index % 2]:
                    row[col] = number_input(
                        col,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                    )

            for index, (col, options) in enumerate(CATEGORY_OPTIONS.items()):
                with input_columns[index % 2]:
                    row[col] = selectbox(col, options)

            submitted = form_submit_button("Predict risk", type="primary")

        if submitted:
            try:
                new_student = DataFrame([row])
                prediction = artifact["model"].predict(new_student)[0]
                probabilities = artifact["model"].predict_proba(new_student)[0]
                label = decode_target(Series([prediction])).iloc[0]

                success(f"Predicted risk level: {label}")

                probability_df = DataFrame(
                    {"probability" : probabilities},
                    index=TARGET_CATEGORIES,
                )

                bar_chart(probability_df, y_label="Probability")
            except Exception as exc:
                error(f"An error occurred: {exc}")
