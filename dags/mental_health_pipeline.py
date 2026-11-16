from datetime import datetime, timedelta
from os import environ
from pathlib import Path
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag

PROJECT_ROOT = Path(environ.get("PROJECT_ROOT", "/opt/airflow/project"))
PIPELINE_COMMAND = f"cd {PROJECT_ROOT} && python -m src.pipeline"

DAG_DOC = """
### Student mental health risk pipeline

Runs the five stages of the project from the command line interface:

| Task | Command | Output |
|---|---|---|
| prepare_data | `python -m src.pipeline prepare` | EDA tables in `reports/eda/`, correlation heatmap |
| preprocess_data | `python -m src.pipeline preprocess` | `data/processed/train.csv` and `test.csv` |
| train_models | `python -m src.pipeline train` | baseline metrics and comparison figure |
| optimize_models | `python -m src.pipeline optimize` | tuning report and `models/optimized_model.joblib` |
| evaluate_model | `python -m src.pipeline evaluate` | final metrics, confusion matrix, `models/final_model.joblib` |

Each task retries twice with a five minute delay. State and logs are available in the Airflow UI.
"""

@dag(
    dag_id="mental_health_pipeline",
    description="EDA, preprocessing, training, tuning and evaluation of the mental health risk model.",
    schedule="0 3 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner" : "brahim",
        "retries" : 2,
        "retry_delay" : timedelta(minutes=5),
        "execution_timeout" : timedelta(minutes=60),
    },
    tags=["mental-health", "machine-learning"],
    doc_md=DAG_DOC,
)
def mental_health_pipeline():
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"{PIPELINE_COMMAND} prepare",
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"{PIPELINE_COMMAND} preprocess",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command=f"{PIPELINE_COMMAND} train",
    )

    optimize_models = BashOperator(
        task_id="optimize_models",
        bash_command=f"{PIPELINE_COMMAND} optimize",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"{PIPELINE_COMMAND} evaluate",
    )

    prepare_data >> preprocess_data >> train_models >> optimize_models >> evaluate_model


dag = mental_health_pipeline()
