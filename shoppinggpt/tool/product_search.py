import sqlite3
from typing import Union, List, Dict

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from shoppinggpt.config import GOOGLE_API_KEY, DATA_PRODUCT_PATH

PRODUCT_RECOMMENDATION_PROMPT = """
You are an expert SQL assistant for an online fashion store database.
Your sole task is to convert the user's natural-language product question into
a valid SQLite SELECT query.

The 'products' table schema:
  product_code   TEXT    — unique product identifier
  product_name   TEXT    — full product name
  material       TEXT    — fabric / material composition
  size           TEXT    — available sizes (e.g. S, M, L, XL)
  color          TEXT    — available colors
  brand          TEXT    — manufacturer or brand name
  gender         TEXT    — target gender (male, female, unisex)
  stock_quantity INTEGER — units currently in stock
  price          REAL    — price in USD

Rules:
• Use LIKE with % for partial/case-insensitive text matching.
• Use LOWER() for case-insensitive comparisons.
• Retrieve all relevant columns (SELECT *) unless a specific column is needed.
• Do NOT include markdown, comments, or explanations — output only the SQL query.

Question: {input}
"""


class ProductDataLoader:
    """Context-manager wrapper around a SQLite connection for product queries."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        if self.conn:
            self.conn.close()

    @staticmethod
    def clean_sql_query(query: str) -> str:
        """Strip any accidental markdown code fences from the LLM output."""
        return query.replace("```sql", "").replace("```", "").strip()

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SQL query and return rows as a list of dicts."""
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        cleaned = self.clean_sql_query(query)
        cursor.execute(cleaned, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


@tool
def product_search_tool(input: str) -> Union[List[Dict], str]:
    """
    Search for products in the store database using natural-language queries.

    The tool converts the user's question into an optimised SQLite query via
    an LLM, executes it against the products database, and returns the matching
    rows as a list of dictionaries.

    Args:
        input: A natural-language product search query
               (e.g. "red cotton shirts under $30 in size M").

    Returns:
        List of product record dicts on success, or an error string on failure.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            temperature=0,
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY
        )
        prompt = PromptTemplate(
            template=PRODUCT_RECOMMENDATION_PROMPT,
            input_variables=["input"]
        )

        with ProductDataLoader(DATA_PRODUCT_PATH) as loader:
            chain = (
                {"input": RunnablePassthrough()}
                | prompt
                | llm
                | (lambda x: loader.execute_query(x.content))
            )
            return chain.invoke(input)

    except Exception as e:
        return f"Product search error: {e}"
