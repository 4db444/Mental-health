from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV
from src.config import CV_FOLDS
from ..feature_engineering import build_pipeline
from .build_models import build_models

def build_param_grids () -> dict[str : dict]:
    return {
        "logistic regression" : {
            "classifier__C" : [0.01, 0.1, 1, 10],
        },
        "random forest" : {
            "classifier__n_estimators" : [200, 400],
            "classifier__max_depth" : [None, 10, 20],
            "classifier__min_samples_leaf" : [1, 4],
        },
        "gradient boosting" : {
            "classifier__n_estimators" : [100, 200],
            "classifier__learning_rate" : [0.05, 0.1],
            "classifier__max_depth" : [3, 5],
        },
    }


def optimize_models (x_train : DataFrame, y_train : Series, use_smote : bool = False) -> list[dict]:
    param_grids = build_param_grids()
    results = []

    for name, classifier in build_models().items():
        pipeline = build_pipeline(classifier, use_smote)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=CV_FOLDS,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        grid_search.fit(x_train, y_train)

        print(f"Optimized: {name} | best f1: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")

        results.append({
            "name" : name,
            "model" : grid_search.best_estimator_,
            "best_params" : grid_search.best_params_,
            "best_score" : grid_search.best_score_,
        })

    return results
