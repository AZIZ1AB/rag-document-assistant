from src.config import (
    DOCUMENTS_DIR, INDEX_PATH, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, STORAGE_DIR,
)
from src.pdf_processing import extract_pdf_pages, create_chunks
from src.vector_store import build_vector_store


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {DOCUMENTS_DIR}")
        print("Add PDF files to data/documents/ and run this script again.")
        return

    pages = []
    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        pages.extend(extract_pdf_pages(pdf_path))

    chunks = create_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print("No selectable text was extracted. Scanned PDFs may need OCR.")
        return

    count = build_vector_store(
        chunks, INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL
    )
    print(f"Index created successfully: {count} chunks")
    print(f"Saved files in: {STORAGE_DIR}")


if __name__ == "__main__":
    main()
