from pandas import DataFrame
from .save_data import save_dataset
from src.config import EDA_REPORT_PATHS, HOURS_IN_WEEK

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
        "total week hours" : df[df["Sleep_Hours"] * 7 + df["Study_Hours_Per_Week"] + df["Screen_Time_Hours"] * 7 >= HOURS_IN_WEEK]
    }

    for key, value in checks_obj.items():
        print(key)
        display(value)

    return checks_obj