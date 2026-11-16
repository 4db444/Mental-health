from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED

def build_models () -> dict[str : object]:
    return {
        "logistic regression" : LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random forest" : RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient boosting" : GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }
