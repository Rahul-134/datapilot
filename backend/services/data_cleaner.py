import pandas as pd
import math

def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}

def clean_dataframe(df: pd.DataFrame, remove_duplicates: bool, handle_nulls: bool, null_strategy: str) -> dict:

    original_rows = df.shape[0]
    dupes_removed = 0
    nulls_handled = 0

    # Step 1 — Remove duplicates
    if remove_duplicates:
        dupes_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()

    # Step 2 — Handle nulls
    if handle_nulls:
        nulls_handled = int(df.isnull().sum().sum())

        if null_strategy == "drop":
            df = df.dropna()

        elif null_strategy == "mean":
            for col in df.select_dtypes(include="number").columns:
                df[col] = df[col].fillna(df[col].mean())
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].fillna("Unknown")

        elif null_strategy == "median":
            for col in df.select_dtypes(include="number").columns:
                df[col] = df[col].fillna(df[col].median())
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].fillna("Unknown")

        elif null_strategy == "unknown":
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].fillna("Unknown")
            for col in df.select_dtypes(include="number").columns:
                df[col] = df[col].fillna(df[col].mean())

    final_rows = df.shape[0]

    return {
        "success": True,
        "original_rows": original_rows,
        "final_rows": final_rows,
        "rows_removed": original_rows - final_rows,
        "dupes_removed": dupes_removed,
        "nulls_handled": nulls_handled,
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample": [sanitize_row(row) for row in df.head(5).to_dict(orient="records")]
    }