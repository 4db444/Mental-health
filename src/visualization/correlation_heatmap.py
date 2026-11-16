import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from src.config import FIGURES_PATH

def plot_correlation (matrix : DataFrame, filename : str = "correlation heatmap.png") -> None:
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title("Feature Correlation (Spearman)")
    plt.tight_layout()

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
