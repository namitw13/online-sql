import os
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv


_APP_DIR = Path(__file__).resolve().parent
_ENV_PATH = _APP_DIR / ".env"
# Streamlit's cwd is often the repo root; load keys from this app's folder.
load_dotenv(_ENV_PATH)
load_dotenv()

st.set_page_config(page_title="AI SQL Data Analyst Agent", page_icon=":bar_chart:", layout="wide")
with st.sidebar:
    st.subheader("API status")
    gem_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
    groq_ok = bool(os.getenv("GROQ_API_KEY", "").strip())
    st.caption("Gemini: " + ("on" if gem_ok else "off (fallback SQL if APIs fail)"))
    st.caption("Groq: " + ("on" if groq_ok else "off"))
st.title("AI SQL Data Analyst Agent")
st.caption("Upload CSV -> Ask questions in plain English -> Get SQL, answer, and chart")

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def build_sqlite_from_csv(dataframe: pd.DataFrame, table_name: str = "uploaded_data") -> str:
    """Create a temporary SQLite DB from a dataframe and return DB path."""
    tmp_dir = tempfile.mkdtemp(prefix="sql_analyst_")
    db_path = str(Path(tmp_dir) / "data.db")
    with sqlite3.connect(db_path) as conn:
        dataframe.to_sql(table_name, conn, index=False, if_exists="replace")
    return db_path


def run_query(sql_text: str, db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql_text, conn)


def choose_chart(df: pd.DataFrame):
    if df.empty:
        return None
    if df.shape[1] < 2:
        return px.histogram(df, x=df.columns[0], title="Auto Chart")
    return px.bar(df, x=df.columns[0], y=df.columns[1], title="Auto Chart")


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split():
        if raw and raw not in COMMON_STOPWORDS:
            tokens.append(raw)
    return tokens


def _best_column_match(question: str, columns: list[str]) -> str | None:
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return None
    best = None
    best_score = 0
    for col in columns:
        col_tokens = set(_tokenize(col))
        if not col_tokens:
            continue
        score = len(q_tokens & col_tokens)
        if score > best_score:
            best_score = score
            best = col
    return best if best_score > 0 else None


def fallback_sql(question: str, columns: list[str], table_name: str = "uploaded_data") -> str:
    q = question.lower().strip()
    col = _best_column_match(question, columns)

    if any(k in q for k in ["count", "how many", "number of", "total rows"]):
        return f"SELECT COUNT(*) AS row_count FROM {table_name};"

    if any(k in q for k in ["distinct", "unique"]) and col:
        return f'SELECT COUNT(DISTINCT "{col}") AS distinct_{col} FROM {table_name};'

    if any(k in q for k in ["average", "avg", "mean"]) and col:
        return f'SELECT AVG("{col}") AS avg_{col} FROM {table_name};'

    if any(k in q for k in ["sum", "total"]) and col:
        return f'SELECT SUM("{col}") AS sum_{col} FROM {table_name};'

    if any(k in q for k in ["max", "highest", "largest"]) and col:
        return f'SELECT MAX("{col}") AS max_{col} FROM {table_name};'

    if any(k in q for k in ["min", "lowest", "smallest"]) and col:
        return f'SELECT MIN("{col}") AS min_{col} FROM {table_name};'

    if any(k in q for k in ["group by", "by "]) and col:
        return (
            f'SELECT "{col}", COUNT(*) AS count FROM {table_name} '
            f'GROUP BY "{col}" ORDER BY count DESC LIMIT 20;'
        )

    # Default exploratory query
    return f"SELECT * FROM {table_name} LIMIT 10;"


def _pandas_dtype_to_sqlite(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TEXT"
    return "TEXT"


def build_schema_context(df: pd.DataFrame, table_name: str = "uploaded_data", sample_rows: int = 2) -> str:
    """Compact schema context to stay within free-tier token limits."""
    lines = [
        f"SQLite table name: {table_name}",
        "Columns (use exactly these names; wrap identifiers in double quotes if needed):",
    ]
    for col in df.columns:
        sqlite_t = _pandas_dtype_to_sqlite(df[col].dtype)
        sample_vals = df[col].dropna().head(2)
        examples = []
        for v in sample_vals:
            s = str(v)
            if len(s) > 40:
                s = s[:37] + "..."
            examples.append(repr(s))
        ex_str = ", ".join(examples) if examples else "(no non-null samples)"
        lines.append(f'  • "{col}" :: {sqlite_t}    sample values: {ex_str}')

    n = min(sample_rows, len(df))
    preview = df.iloc[:n]
    try:
        rows_json = preview.to_json(orient="records", default_handler=str)
    except Exception:
        rows_json = preview.astype(str).to_json(orient="records")
    if len(rows_json) > 3500:
        rows_json = rows_json[:3497] + "..."
    lines.append("")
    lines.append(f"First {n} rows as JSON:")
    lines.append(rows_json)
    return "\n".join(lines)


def _strip_sql_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _gemini_models_to_try() -> list[str]:
    preferred = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"
    fallbacks = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]
    out: list[str] = []
    for m in [preferred] + fallbacks:
        if m and m not in out:
            out.append(m)
    return out


def _extract_gemini_text(resp) -> str:
    try:
        return (resp.text or "").strip()
    except ValueError:
        raw_out = ""
        if getattr(resp, "candidates", None):
            for c in resp.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            raw_out = p.text.strip()
                            break
                if raw_out:
                    break
        return raw_out


def generate_sql_gemini(question: str, schema_context: str, table_name: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    instructions = f"""You are an expert SQLite analyst. The user's data is in ONE table only.

Rules:
- Output exactly ONE valid SQLite SELECT query (no INSERT/UPDATE/DELETE). End with semicolon if natural.
- Table name is exactly: {table_name}
- Prefer choosing specific columns named in the question instead of SELECT * when possible.
- Quote column names with double quotes when they contain spaces or special characters: example SELECT "Sales Q1" FROM ...
- For filtering text by partial match use LIKE with wildcards, e.g. WHERE LOWER("name") LIKE '%alice%' 
- For "top N", "highest", "lowest" use ORDER BY ... LIMIT N.
- For grouping ("by region", "per category") use GROUP BY and aggregates (COUNT, SUM, AVG, MIN, MAX).
- For dates stored as TEXT use SQLite date functions only if the column looks like dates; otherwise compare as text.
- Use CAST("col" AS REAL) when averaging/summing text-encoded numbers if needed.
- If the question is ambiguous, pick the most reasonable interpretation from the schema.

Respond with ONLY the SQL query. No markdown fences, no explanation."""

    user_block = f"""Schema and samples:
{schema_context}

User question:
{question}"""

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    full_prompt = f"{instructions}\n\n{user_block}"

    for model_name in _gemini_models_to_try():
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                full_prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            )
            raw_out = _extract_gemini_text(resp)
            if raw_out:
                return _strip_sql_fences(raw_out)
            continue
        except Exception as err:
            err_s = str(err).lower()
            if any(x in err_s for x in ("429", "quota", "resource exhausted", "rate", "limit: 0")):
                continue
            if "404" in str(err) or "not found" in err_s:
                continue
            return None

    return None


def generate_sql_groq(question: str, schema_context: str, table_name: str) -> str | None:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        return None

    prompt = f"""You write ONE SQLite SELECT query for this database.

{schema_context}

Table name: {table_name}

Question: {question}

Rules: Only SELECT; quote identifiers with double quotes when needed; use LIKE for partial text matches;
GROUP BY for breakdowns; ORDER BY + LIMIT for top/bottom.

Output ONLY the SQL query. No markdown."""
    preferred_groq = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
    groq_try = [preferred_groq, "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    groq_models: list[str] = []
    for m in groq_try:
        if m and m not in groq_models:
            groq_models.append(m)

    for groq_model in groq_models:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": groq_model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            if response.status_code in (400, 404):
                continue
            if not response.ok:
                return None
            raw = response.json()["choices"][0]["message"]["content"].strip()
            cleaned = _strip_sql_fences(raw)
            if cleaned:
                return cleaned
        except Exception:
            return None

    return None


def generate_sql_with_llm(
    question: str,
    df: pd.DataFrame,
    table_name: str = "uploaded_data",
) -> str:
    columns = list(df.columns)
    schema_context = build_schema_context(df, table_name=table_name)

    sql = generate_sql_gemini(question, schema_context, table_name)
    if sql:
        return sql

    sql = generate_sql_groq(question, schema_context, table_name)
    if sql:
        return sql

    return fallback_sql(question, columns, table_name=table_name)


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("CSV Preview")
    st.dataframe(df.head(20), width="stretch")

    db_path = build_sqlite_from_csv(df, table_name="uploaded_data")

    question = st.text_input("Ask a question about your data")
    run_btn = st.button("Analyze")

    if run_btn and question.strip():
        with st.spinner("Thinking..."):
            suggested_sql = generate_sql_with_llm(question, df, table_name="uploaded_data")
            sql_query = st.text_area(
                "SQL to execute (you can edit before running)",
                value=suggested_sql,
                height=120,
            ).strip()
            try:
                result_df = run_query(sql_query, db_path)
                answer = (
                    f"Query returned {len(result_df)} row(s)."
                    if not result_df.empty
                    else "No rows matched your question."
                )
            except Exception as query_err:
                result_df = pd.DataFrame()
                answer = f"Could not run generated SQL. Error: {query_err}"

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Generated SQL")
            st.code(sql_query, language="sql")

            st.subheader("Result")
            st.dataframe(result_df, width="stretch")

            fig = choose_chart(result_df)
            if fig is not None:
                st.subheader("Visualization")
                st.plotly_chart(fig, width="stretch")

else:
    st.info("Upload a CSV to begin.")
