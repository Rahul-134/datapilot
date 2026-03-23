import os
import re
import math
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}

def build_prompt(user_query: str, df: pd.DataFrame) -> str:
    col_info = "\n".join(
        f"  - {col} ({df[col].dtype})" for col in df.columns
    )
    sample = df.head(3).to_string(index=False)

    return f"""You are a Python/pandas expert. A user has uploaded a DataFrame called `df`.

DataFrame Info:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Columns:
{col_info}

Sample Data:
{sample}

User Request: "{user_query}"

Your job:
1. Write a single Python expression or short code block using pandas that answers the user's request.
2. The result MUST be stored in a variable called `result`.
3. `result` must be either a pandas DataFrame or a pandas Series.
4. Do NOT import anything. Do NOT redefine `df`. Only use pandas operations on `df`.
5. Do NOT include any explanation — return ONLY the Python code, no markdown, no backticks.

Example outputs:
result = df[df['age'] > 30]
result = df.groupby('category')['sales'].sum()
result = df.sort_values('price', ascending=False).head(10)
"""

def execute_code(code: str, df: pd.DataFrame) -> dict:
    # Strip markdown fences if model adds them anyway
    code = re.sub(r"```(?:python)?|```", "", code).strip()

    local_vars = {"df": df.copy(), "pd": pd}

    try:
        exec(code, {"__builtins__": {}}, local_vars)
    except Exception as e:
        return {"error": f"Code execution failed: {str(e)}", "generated_code": code}

    result = local_vars.get("result")

    if result is None:
        return {"error": "Model did not produce a `result` variable.", "generated_code": code}

    if not isinstance(result, (pd.DataFrame, pd.Series)):
        return {"error": f"Result must be a DataFrame or Series, got {type(result).__name__}", "generated_code": code}

    # Convert Series to DataFrame
    if isinstance(result, pd.Series):
        result = result.reset_index()
        result.columns = [str(c) for c in result.columns]

    rows    = [sanitize_row(row) for row in result.head(100).to_dict(orient="records")]
    columns = result.columns.tolist()

    return {
        "success":        True,
        "generated_code": code,
        "columns":        columns,
        "rows":           rows,
        "total_rows":     len(result),
        "shape":          list(result.shape)
    }

def query_dataframe(user_query: str, df: pd.DataFrame) -> dict:
    prompt = build_prompt(user_query, df)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        code = response.text.strip()
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

    return execute_code(code, df)