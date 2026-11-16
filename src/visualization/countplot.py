import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

def countplot(df : DataFrame) -> None:
    plt.title("Distribution of Risk Levels")
    sns.countplot(
        data=df,
        x="Mental_Health_Risk"
    )

    plt.show()
