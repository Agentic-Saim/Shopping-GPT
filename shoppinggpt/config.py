# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ── Base directory (repo root, two levels up from this file) ──────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Paths ─────────────────────────────────────────────────────────────────
DATA_PRODUCT_PATH = str(BASE_DIR / "data" / "products.db")
DATA_TEXT_PATH    = str(BASE_DIR / "data" / "policy.txt")
STORE_DIRECTORY   = str(BASE_DIR / "data" / "datastore")

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")