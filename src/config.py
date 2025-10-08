import os

# Embeddings / DB
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "hf"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if needed

# Safety: make 'openai' optional; fall back to HF if no API key
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
if not HAS_OPENAI_KEY:
    LLM_PROVIDER = "hf"

# Directory where the Chroma vector DB will be stored
VECTOR_DB_DIR = "chroma"
