from pandas import DataFrame, Series
from ..feature_engineering import build_pipeline
from .build_models import build_models

def train_models (x_train : DataFrame, y_train : Series, use_smote : bool = False) -> list[dict]:
    results = []

    for name, classifier in build_models().items():
        pipeline = build_pipeline(classifier, use_smote)
        pipeline.fit(x_train, y_train)

        print(f"Trained: {name}")

        results.append({
            "name" : name,
            "model" : pipeline,
        })

    return results
