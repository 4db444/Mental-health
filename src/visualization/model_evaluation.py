import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from pandas import DataFrame
from src.config import FIGURES_PATH, TARGET_CATEGORIES

def plot_confusion_matrix (matrix : ndarray, model_name : str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_CATEGORIES,
        yticklabels=TARGET_CATEGORIES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_PATH / f"confusion matrix - {model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_model_comparison (report : DataFrame, metric : str = "f1", filename : str = "model comparison.png") -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=report.index, y=report[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xticks(rotation=15)
    plt.tight_layout()

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
