# Student Mental Health Risk Prediction

Predict the mental-health risk level of a university student (`Low`, `Medium`, `High`) from academic, lifestyle and psychosocial survey features.

The dataset is synthetic: 2,000 students, 16 input features and one three-class target. Full column documentation lives in [`docs/data explination.md`](docs/data%20explination.md).

## Project Structure

```
src/
  config.py                       # paths, column lists, seed, thresholds
  data/
    load_data.py                  # CSV loading
    save_data.py                  # CSV artefact writer
    split_data.py                 # stratified 80/20 split
    eda.py                        # missing values, duplicates, inconsistencies, outliers,
                                  # descriptive stats, target balance, feature/target links, correlation
  feature_engineering/
    build_pipeline.py             # imputation + encoding + scaling + optional SMOTE
    target.py                     # Low/Medium/High <-> 0/1/2
  train/
    build_models.py               # logistic regression, random forest, gradient boosting
    train.py                      # baseline training
    optimize.py                   # GridSearchCV tuning
    evaluate.py                   # metrics, reports, error analysis, joblib export
  pipeline/
    stages.py                     # prepare, preprocess, train, optimize, evaluate stages
    run.py                        # stage registry and run_pipeline
    __main__.py                   # python -m src.pipeline <stage>
  visualization/                  # countplot, boxplots, correlation heatmap, confusion matrix
app/
  streamlit_app.py                # performance dashboard + prediction form
dags/
  mental_health_pipeline.py       # Airflow DAG calling the pipeline stages
notebooks/
  01_EDA.ipynb                    # exploration only
  02_modeling.ipynb               # preprocessing, training, tuning, final evaluation
reports/
  eda/                            # EDA tables
  models/                         # metrics, classification reports, errors
figures/                          # heatmaps and confusion matrices
models/final_model.joblib         # preprocessing + classifier artefact
docker/airflow/Dockerfile         # Airflow image with the project dependencies
docker/streamlit/Dockerfile       # Streamlit image with the project dependencies
docker-compose.yml                # Airflow + Postgres + Streamlit
```

Logic lives in small functions under `src/`; notebooks only orchestrate.

## Workflow

### 1. Exploratory analysis (`notebooks/01_EDA.ipynb`)

- Structure, dtypes and descriptive statistics.
- Missing values, duplicate rows, inconsistencies and IQR outliers.
- Target distribution and class balance (660 / 660 / 680, ratio 1.03).
- Numerical and categorical features grouped by risk level.
- Spearman correlation matrix after encoding:
  - ordinal: `Physical_Activity`, `Diet_Quality`, target;
  - binary: `Part_Time_Job`, `Previous_Mental_Health_History`;
  - nominal: `Gender` one-hot encoded.

Outputs are written to `reports/eda/` and `figures/correlation heatmap.png`.

### 2. Preprocessing (`notebooks/02_modeling.ipynb`)

The raw data is split 80/20 with stratification before any fitting.

| Decision | Choice | Justification |
|---|---|---|
| Missing numerical values | median imputation | robust to skew and outliers |
| Missing categorical values | most frequent | keeps the dominant category |
| Ordinal features | ordered encoding | Low/Moderate/High and Poor/Average/Good carry order |
| Binary features | No/Yes -> 0/1 | natural binary encoding |
| Nominal feature | one-hot, first level dropped | no natural order; avoids collinearity |
| Outliers | kept, `RobustScaler` | the dataset documents them as realistic edge cases |
| Class imbalance | checked, SMOTE only above 1.5 ratio | current ratio is 1.03, so no resampling is required |
| Leakage | every transform inside the pipeline | fitted on training folds only |

### 3. Models and tuning

Three classifiers are compared on identical training data:

- logistic regression (scaled baseline, `class_weight="balanced"`);
- random forest (non-linearity, feature interactions);
- gradient boosting (strong tabular baseline).

Metrics: accuracy, weighted precision/recall/F1, macro one-vs-rest ROC-AUC and confusion matrix. Hyperparameters are tuned with `GridSearchCV` (5 folds, weighted F1). The best pipeline is evaluated once on the held-out test set, then exported with `joblib`.

### 4. Orchestration with Airflow

`dags/mental_health_pipeline.py` chains five tasks, one per pipeline stage:

```
prepare_data >> preprocess_data >> train_models >> optimize_models >> evaluate_model
```

Each task runs `python -m src.pipeline <stage>` from the project root, so the DAG has no
business logic of its own and every stage can be reproduced by hand:

| Stage | Command | Output |
|---|---|---|
| prepare | `python -m src.pipeline prepare` | EDA tables, correlation heatmap |
| preprocess | `python -m src.pipeline preprocess` | `data/processed/train.csv`, `test.csv` |
| train | `python -m src.pipeline train` | baseline metrics, comparison figure |
| optimize | `python -m src.pipeline optimize` | tuning report, `models/optimized_model.joblib` |
| evaluate | `python -m src.pipeline evaluate` | final metrics, confusion matrix, `models/final_model.joblib` |

The DAG is scheduled weekly (`0 3 * * 1`), does not backfill (`catchup=False`), and each
task retries twice after five minutes. Task state, duration, logs and failures are
available in the Airflow UI. `python -m src.pipeline all` runs the same stages in order
without Airflow.

### 5. Streamlit application

`app/streamlit_app.py` loads `models/final_model.joblib` once and provides two tabs:

- **Model Performance** — final metrics, confusion matrix, classification report,
  optimization report, correlation heatmap and model comparison;
- **Predict Risk** — a form with the ten numerical and five categorical features that
  returns the predicted risk level and the class probabilities.

The app reads only saved artefacts; it never retrains.

### 6. Docker stack

`docker-compose.yml` runs three services:

| Service | Image | Purpose | Port |
|---|---|---|---|
| `postgres` | `postgres:16` | Airflow metadata database | — |
| `airflow` | `mental-health-airflow:local` | scheduler, API server, DAG runs | 8080 |
| `streamlit` | `mental-health-streamlit:local` | performance dashboard and prediction form | 8501 |

Both application images are built from the project (`docker/airflow/Dockerfile`,
`docker/streamlit/Dockerfile`): the build installs the project in editable mode from
`pyproject.toml` and `src/`, so dependencies are baked into the image once instead of
being pip-installed on every container start. The compose file still mounts `./` into the
container, so code and artefacts edited on the host are picked up without a rebuild.

`airflow` and `streamlit` share the same artifacts on disk: the DAG writes the metrics,
figures and models, and the app reads them.

## Run

```bash
uv sync
```

Run the whole pipeline from the command line (same stages as the DAG):

```bash
uv run python -m src.pipeline all
```

Or run one stage at a time: `prepare`, `preprocess`, `train`, `optimize`, `evaluate`.

Start the Streamlit application:

```bash
uv run streamlit run app/streamlit_app.py
```

Start Airflow and Streamlit locally with Docker Compose:

```bash
cp .env.example .env
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
docker compose up --build
```

- Airflow UI: `http://localhost:8080`. `airflow standalone` creates an `admin` user and
  prints the generated password in `docker compose logs airflow` (search for `password`).
- Streamlit: `http://localhost:8501`.

`AIRFLOW_UID` makes the reports, figures and models written by the DAG belong to your host
user, so the Streamlit container and your editor can read them.

Airflow's own logs live in the `airflow-logs` named volume, not in a `./logs` bind mount.
Docker seeds a named volume from the image, so the folder keeps the image's
`airflow:0` ownership and stays writable whatever `AIRFLOW_UID` is. This matters because
Airflow 3's dag processor exits if it cannot write its parse logs, which leaves the UI
running but with an empty DAGs page.

The DAG is `mental_health_pipeline` and is unpaused on creation, so the next weekly run
happens on its own. Trigger it once after the first start:

```bash
docker compose exec airflow airflow dags trigger mental_health_pipeline
```

The Streamlit service shows a warning until the pipeline has produced
`reports/models/` and `models/final_model.joblib`.

Rebuild the images only when the dependency list in `pyproject.toml` changes:

```bash
docker compose build
```

The compose file mounts the repository into both containers, so changes to `src/`,
`dags/` or `app/` do not require a rebuild.

If the DAGs page stays empty, the dag processor is not running. Check
`docker compose logs airflow | grep -iE "error|permission|dag"`; `docker compose down -v`
recreates the metadata and log volumes from scratch.

A quick smoke check without notebooks:

```bash
uv run python -c "from src import laod_dataset, split_dataset; from src.config import RAW_DATA_PATH, RAW_DATA_FILE; df = laod_dataset(RAW_DATA_PATH, RAW_DATA_FILE); print(split_dataset(df)[0].shape)"
```

Artifacts:

- `data/processed/train.csv` and `test.csv` — stratified 80/20 split used by the DAG stages.
- `models/optimized_model.joblib` — best tuned pipeline, intermediate artefact of the DAG.
- `models/final_model.joblib` — dict with `model` (full pipeline), `best_model_name`, `feature_columns`, `target_categories`.
- `reports/models/` — baseline metrics, optimization report, final metrics/report/confusion matrix, misclassified rows.
- `figures/` — correlation heatmap, model comparisons and confusion matrix.

## Results

First full run (seed 42, 80/20 stratified split, 5-fold CV):

| Model | CV weighted F1 | Selected |
|---|---|---|
| logistic regression | 0.7476 | yes (`C=10`) |
| random forest | 0.6883 | no |
| gradient boosting | 0.6986 | no |

Final held-out test metrics for the selected pipeline:

| Metric | Value |
|---|---|
| accuracy | 0.7575 |
| weighted precision | 0.7584 |
| weighted recall | 0.7575 |
| weighted F1 | 0.7579 |
| macro OVR ROC-AUC | 0.9142 |

Per-class F1: Low 0.8168, Medium 0.6391, High 0.8162. All 97 misclassifications are between adjacent risk levels (Low <-> Medium or Medium <-> High); there are no Low <-> High errors.

Logistic regression wins because the synthetic target is roughly an additive weighted score and the relationships are mostly monotonic and near-linear. The tree models overfit the training rows (random forest reaches 1.00 training accuracy) and generalize worse under cross-validation.

Source of truth: `reports/models/final metrics.csv`, `reports/models/final classification report.csv`, `reports/models/final confusion matrix.csv` and `reports/models/errors - logistic regression.csv`.

## Limits

- The target is derived from a weighted synthetic score, so high scores mostly confirm the documented generation logic.
- Correlation is not causation; the model predicts a risk band, not a diagnosis.
- The model should support human decision-making, never replace it. Errors between adjacent risk levels are the expected failure mode.
