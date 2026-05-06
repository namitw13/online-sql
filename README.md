# AI SQL Data Analyst Agent (CSV -> SQL -> Insights)

## Objective
This Streamlit app uploads a CSV, converts it to SQLite, accepts natural language questions, generates SQL using Groq LLM, executes the SQL, and returns:
- Answer
- SQL query
- Visualization

## Tech Stack
- Frontend: Streamlit
- LLM: Groq (Llama 3)
- Framework: LangChain
- Database: SQLite
- Data: Pandas
- Visualization: Plotly

## Project Structure
- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `.env.example` - environment variable template

## Setup
1. Create and activate virtual environment:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your `GROQ_API_KEY`
4. Run app:
   - `streamlit run app.py`

## How It Works
1. Upload a CSV file.
2. App writes CSV content to SQLite table: `uploaded_data`.
3. Ask a natural language question.
4. Groq model generates SQL query.
5. SQL executes against SQLite.
6. App displays answer, SQL, result table, and chart.

## Notes
- If Groq key is missing, SQL generation falls back to a default query.
- The "Advanced Agent Response" uses a LangChain SQL agent for additional reasoning.
