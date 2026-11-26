import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
STORAGE_DIR = ROOT_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "rag_index.faiss"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"

EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
GENERATION_MODEL = os.getenv(
    "RAG_GENERATION_MODEL",
    "Qwen/Qwen2.5-3B-Instruct",
)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
