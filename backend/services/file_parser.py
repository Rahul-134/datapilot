import pandas as pd
import io
import math

# In-memory store — holds current session DataFrame
_store: dict = {"df": None, "filename": None}

def get_stored_df():
    return _store["df"]

def get_stored_filename():
    return _store["filename"]

def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}

async def parse_file(file):
    contents = await file.read()

    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return {"error": "Unsupported file format"}

    _store["df"]       = df.copy()
    _store["filename"] = file.filename

    overview = {
        "filename":       file.filename,
        "rows":           df.shape[0],
        "columns":        df.shape[1],
        "column_names":   df.columns.tolist(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "null_counts":    df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample":         [sanitize_row(row) for row in df.head(5).to_dict(orient="records")]
    }

    return overview