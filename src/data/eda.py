from pandas import DataFrame, crosstab, get_dummies
from .save_data import save_dataset
from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EDA_REPORT_PATHS,
    HOURS_IN_WEEK,
    ID_COL,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    TARGET_CATEGORIES,
    TARGET_COL,
    TARGET_MAP,
)
from src.utils import numerical_cols

try:
    from IPython.display import display
except ImportError:
    display = print

def missing_values (df : DataFrame) -> None: 
    na_sum = df.isna().sum()[df.isna().sum() > 0]

    missing_df = DataFrame({
        "columns" : na_sum.keys(),
        "missing values" : na_sum.values,
        "percentage" : ((na_sum / len(df)) * 100).values
    })

    missing_df = missing_df.set_index("columns")
    
    print("Missing Values State.")
    print(missing_df)

    save_dataset(missing_df, EDA_REPORT_PATHS, "missing value state.csv", True)


def duplicate_values (df : DataFrame) -> None:
    dup_df = DataFrame({
        "Duplicated rows count" : [df.duplicated().sum()]
    })

    print("Duplicated Rows State.")
    print(dup_df)

    save_dataset(dup_df, EDA_REPORT_PATHS, "duplicated rows state.csv")

def inconsistencies (df : DataFrame) -> dict[str : DataFrame]:
    checks_obj = {
        "socialmedia over screen time" : df[df["Social_Media_Hours"] > df["Screen_Time_Hours"]],
        "non-binary" : df[df["Gender"] == "Non-binary"],
        "total week hours" : df[df["Sleep_Hours"] * 7 + df["Study_Hours_Per_Week"] + df["Screen_Time_Hours"] * 7 >= HOURS_IN_WEEK],
        "age" : df[df["Age"] <=0],
        "gpa" : df[(df["GPA"] > 4) | (df["GPA"] < 0)]
    }

    for key, value in checks_obj.items():
        if len(value) :
            save_dataset(value, EDA_REPORT_PATHS, f"Inconsistincy - {key}.csv")

            print(key)
            display(value)

    return checks_obj

def outliers (df : DataFrame) -> None:
    num_cols = numerical_cols(df)

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = round(Q1 - 1.5 * IQR, 2)
        upper_bound = round(Q3 + 1.5 * IQR, 2)

        df_outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]

        if(len(df_outliers)) : 
            print("Column: ", col)
            print("Upper bound: ", upper_bound)
            print("Lower bound", lower_bound)

            display(df_outliers)
            save_dataset(df_outliers, EDA_REPORT_PATHS, f"Outliers - {col}.csv")


def descriptive_statistics (df : DataFrame) -> DataFrame:
    stats_df = df.drop(columns=[ID_COL]).describe(include="all").T

    save_dataset(stats_df, EDA_REPORT_PATHS, "descriptive statistics.csv", True)

    return stats_df


def target_distribution (df : DataFrame, target_col : str = TARGET_COL) -> DataFrame:
    counts = df[target_col].value_counts().reindex(TARGET_CATEGORIES)

    distribution_df = DataFrame({
        "count" : counts.values,
        "percentage" : (counts / len(df) * 100).round(2).values,
        "ratio to smallest class" : (counts / counts.min()).round(3).values,
    }, index=counts.index)

    print("Target Distribution.")
    print(distribution_df.to_string())

    save_dataset(distribution_df, EDA_REPORT_PATHS, "target distribution.csv", True)

    return distribution_df


def numerical_target_summary (df : DataFrame, target_col : str = TARGET_COL) -> DataFrame:
    summary_df = df.groupby(target_col)[NUMERICAL_FEATURES].mean().round(3).T

    print("Numerical Features by Target.")
    print(summary_df.to_string())

    save_dataset(summary_df, EDA_REPORT_PATHS, "numerical features by target.csv", True)

    return summary_df


def categorical_target_summary (df : DataFrame, target_col : str = TARGET_COL) -> dict[str : DataFrame]:
    summary_obj = {}

    for col in CATEGORICAL_FEATURES:
        table = crosstab(df[col], df[target_col], normalize="index").round(3)
        summary_obj[col] = table

        print(f"Categorical Feature by Target: {col}")
        print(table.to_string())

        save_dataset(table, EDA_REPORT_PATHS, f"categorical by target - {col}.csv", True)

    return summary_obj


def _encode_for_correlation (df : DataFrame) -> DataFrame:
    corr_df = df.drop(columns=[ID_COL]).copy()

    for col, categories in ORDINAL_FEATURES.items():
        corr_df[col] = corr_df[col].map({category : index for index, category in enumerate(categories)})

    for col in BINARY_FEATURES:
        corr_df[col] = corr_df[col].map({"No" : 0, "Yes" : 1})

    corr_df[TARGET_COL] = corr_df[TARGET_COL].map(TARGET_MAP)
    corr_df = get_dummies(corr_df, columns=NOMINAL_FEATURES, drop_first=True, dtype=int)

    return corr_df


def correlation (df : DataFrame, method : str = "spearman") -> DataFrame:
    matrix = _encode_for_correlation(df).corr(method=method)

    target_correlation = matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)

    print(f"Correlation Matrix ({method}).")
    print(matrix.round(2).to_string())

    print(f"Correlation with {TARGET_COL}.")
    print(target_correlation.round(3).to_string())

    save_dataset(matrix, EDA_REPORT_PATHS, f"correlation matrix - {method}.csv", True)
    save_dataset(target_correlation.to_frame("correlation"), EDA_REPORT_PATHS, f"correlation with target - {method}.csv", True)

    return matrix
