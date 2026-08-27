import pandas as pd
import io
import math
from backend.services.session_store import get_value, set_value


def _df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def _csv_to_df(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text))


def get_stored_df(session_id: str):
    csv_text = get_value(session_id, "df")
    return _csv_to_df(csv_text) if csv_text is not None else None


def get_stored_filename(session_id: str):
    return get_value(session_id, "filename")


def set_stored_df(session_id: str, df: pd.DataFrame):
    """Called after cleaning to update the active DataFrame."""
    set_value(session_id, "df", _df_to_csv(df))


def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}


async def parse_file(session_id: str, file):
    contents = await file.read()

    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return {"error": "Unsupported file format"}

    set_value(session_id, "df", _df_to_csv(df))
    set_value(session_id, "filename", file.filename)

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
