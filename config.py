import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = "chroma_db"  # Path to store ChromaDB persistent data
COLLECTION_NAME = "my_rag_documents"
EMBEDDING_MODEL_NAME = "gemini-embedding-001" # Recommended Gemini embedding model
GENERATION_MODEL_NAME = "gemini-1.5-flash" # Or "gemini-1.5-flash"