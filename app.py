from pathlib import Path
import tempfile

import streamlit as st

from src.config import (
    DOCUMENTS_DIR, INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL,
    GENERATION_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)
from src.pdf_processing import extract_pdf_pages, create_chunks
from src.vector_store import (
    build_vector_store, load_vector_store, load_embedding_model, retrieve,
)
from src.generator import LocalAnswerGenerator

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption("Ask questions about your PDFs and inspect the passages behind each answer.")

with st.sidebar:
    st.header("Configuration")
    embedding_name = st.text_input("Embedding model", value=EMBEDDING_MODEL)
    generation_name = st.text_input("Generation model", value=GENERATION_MODEL)
    chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=5000, value=CHUNK_SIZE, step=50)
    overlap = st.number_input("Chunk overlap", min_value=0, max_value=4999, value=CHUNK_OVERLAP, step=25)
    top_k = st.slider("Retrieved passages (top-k)", min_value=1, max_value=10, value=TOP_K)
    st.caption("The first run downloads model weights. Local generation may be slow on CPU.")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Build / Update index", type="primary", disabled=not uploaded_files):
    if overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        all_pages = []
        progress = st.progress(0)
        status = st.empty()

        for i, uploaded in enumerate(uploaded_files):
            safe_name = Path(uploaded.name).name
            destination = DOCUMENTS_DIR / safe_name
            destination.write_bytes(uploaded.getvalue())
            status.write(f"Extracting {safe_name}...")
            all_pages.extend(extract_pdf_pages(destination))
            progress.progress((i + 1) / len(uploaded_files))

        chunks = create_chunks(all_pages, int(chunk_size), int(overlap))
        if not chunks:
            st.error("No selectable text was extracted. The PDFs may be scanned images.")
        else:
            with st.spinner("Creating embeddings and building the FAISS index..."):
                count = build_vector_store(
                    chunks, INDEX_PATH, CHUNKS_PATH, embedding_name
                )
            st.success(f"Index built: {count} text chunks from {len(all_pages)} pages.")
            st.cache_resource.clear()

@st.cache_resource
def get_embedding_model(model_name):
    return load_embedding_model(model_name)

@st.cache_resource
def get_generator(model_name):
    return LocalAnswerGenerator(model_name)

index_exists = INDEX_PATH.exists() and CHUNKS_PATH.exists()

if not index_exists:
    st.info("Upload PDF files and build an index to start asking questions.")
else:
    try:
        index, chunks = load_vector_store(INDEX_PATH, CHUNKS_PATH)
        st.success(f"Index ready — {len(chunks)} chunks.")
        question = st.chat_input("Ask a question about your documents...")

        if question:
            with st.spinner("Searching for relevant passages..."):
                embedding_model = get_embedding_model(embedding_name)
                results = retrieve(
                    question, index, chunks, embedding_model, int(top_k)
                )

            with st.expander("Retrieved sources", expanded=True):
                if results:
                    for number, item in enumerate(results, start=1):
                        st.markdown(
                            f"**[{number}] {item['source']} — page {item['page']}**  \n"
                            f"Similarity: `{item['score']:.4f}`"
                        )
                        st.write(item["text"])
                        st.divider()
                else:
                    st.write("No passages were retrieved.")

            with st.spinner("Generating an answer with the local language model..."):
                generator = get_generator(generation_name)
                try:
                    answer = generator.answer(question, results)
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as error:
                    st.error(f"Generation failed: {error}")
                    st.info("Check the model ID, available memory, and internet connection.")
    except Exception as error:
        st.error(f"Could not load the index: {error}")
        st.warning("Rebuild the index to ensure the FAISS index and metadata match.")
