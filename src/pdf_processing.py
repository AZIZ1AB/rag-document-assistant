from pathlib import Path
import fitz


def extract_pdf_pages(pdf_path):
    """Extract non-empty text pages and preserve source filename/page number."""
    pdf_path = Path(pdf_path)
    extracted_pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                extracted_pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                })

    return extracted_pages


def split_into_chunks(text, chunk_size=900, overlap=150):
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(pages, chunk_size=900, overlap=150):
    """Create chunk records with stable IDs and source metadata."""
    all_chunks = []

    for page in pages:
        page_chunks = split_into_chunks(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk_id, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "id": len(all_chunks),
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_id,
            })

    return all_chunks
