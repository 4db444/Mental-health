import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from src.utils import numerical_cols
import math

def boxplot(df : DataFrame, x_col : str, y_col : str) -> None:
    plt.title("Distribution of Risk Levels")
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col
    )

    plt.show()

def full_boxplot_state(df: DataFrame, x_col) -> None:
    num_cols = numerical_cols(df)

    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows)
    )

    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(
            data=df,
            x=x_col,
            y=col,
            ax=axes[i]
        )
        axes[i].set_title(f"Distribution of {col}")

    for i in range(len(num_cols), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Distribution of Numerical Variables by Risk Level", fontsize=16)

    fig.tight_layout()
    plt.show()