import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


def load_embedding_model(model_name=EMBEDDING_MODEL):
    return SentenceTransformer(model_name)


def build_vector_store(chunks, index_path, chunks_path, model_name=EMBEDDING_MODEL):
    """Embed chunks and persist a FAISS index plus aligned metadata."""
    if not chunks:
        raise ValueError("No text chunks were found. Check your PDF files.")

    model = load_embedding_model(model_name)
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = Path(index_path)
    chunks_path = Path(chunks_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with chunks_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    return len(chunks)


def load_vector_store(index_path, chunks_path):
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Index not found. Build the document index first.")

    index = faiss.read_index(str(index_path))
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunks metadata do not match. Rebuild the index."
        )

    return index, chunks


def retrieve(question, index, chunks, embedding_model, top_k=4):
    """Return the top-k passages with similarity scores and metadata."""
    if not question.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, positions = index.search(query_vector, min(top_k, index.ntotal))
    results = []

    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue
        item = dict(chunks[int(position)])
        item["score"] = float(score)
        results.append(item)

    return results
