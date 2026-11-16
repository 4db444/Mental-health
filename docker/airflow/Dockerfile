FROM apache/airflow:3.4.0-python3.13

COPY --chown=airflow:root pyproject.toml /opt/airflow/project/pyproject.toml
COPY --chown=airflow:root src /opt/airflow/project/src

RUN pip install --no-cache-dir -e /opt/airflow/project
