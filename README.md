# RAG Document Assistant

A practical Retrieval-Augmented Generation (RAG) project built with **Python, Streamlit, PyMuPDF, Sentence Transformers, FAISS, and Hugging Face Transformers**.

Upload PDF documents, build a searchable vector index, ask questions, and inspect the source passages (filename and page number) used to prepare each answer.

## Features

- Extracts text page-by-page from PDF files.
- Splits extracted text into overlapping chunks while preserving source, page, and chunk metadata.
- Creates normalized multilingual sentence embeddings.
- Uses FAISS inner-product search (cosine similarity for normalized vectors).
- Generates grounded answers with a local Hugging Face instruction model.
- Displays retrieved passages and similarity scores for verification.
- Includes basic retrieval metric functions and unit tests.

## Architecture

```text
PDF files
   |
   v
PyMuPDF extraction (page + source metadata)
   |
   v
Text chunking (900 characters, 150 overlap by default)
   |
   v
Sentence Transformer embeddings
   |
   v
FAISS vector index + chunks.json
   |
User question
   |
   v
Question embedding -> FAISS top-k retrieval
   |
   v
Context + question -> Hugging Face language model
   |
   v
Answer + source passages
```

## Requirements

- Python 3.10 or newer
- Internet connection for the first run (to install packages and download model weights)
- Enough RAM/disk space for the selected models

The default generation model is `Qwen/Qwen2.5-0.5B-Instruct`, selected as a lighter starting point than the 3B model discussed in the accompanying teaching guide. It can still be slow on a CPU and model performance depends on your hardware. The embedding model is multilingual.

> **Hardware note:** The language model and embedding model are downloaded on first use. If your computer has limited RAM, start with a small document collection and expect indexing/generation to take time. You can change model IDs in the sidebar or environment variables.

## Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/YOUR-USERNAME/rag-document-assistant.git
cd rag-document-assistant

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
git clone https://github.com/YOUR-USERNAME/rag-document-assistant.git
cd rag-document-assistant

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a GPU-specific PyTorch installation, use the official selector:
https://pytorch.org/get-started/locally/

## Run the application

```bash
streamlit run app.py
```

1. Open the local URL printed in the terminal (usually `http://localhost:8501`).
2. Upload one or more text-based PDF files.
3. Click **Build / Update index**.
4. Ask a question in the chat input.
5. Expand **Retrieved sources** to inspect the passages, pages, and scores.

The first indexing run downloads the embedding model. The first question that requires generation downloads the language model.

## Optional: build the index from a folder

Put PDF files in `data/documents/`, then run:

```bash
python build_index.py
```

The generated index is saved in `storage/`:
- `rag_index.faiss` — FAISS vector index
- `chunks.json` — chunk text and source metadata

These generated files are intentionally excluded from Git. Rebuild them from your own PDFs.

## Evaluate retrieval metrics

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

`src/evaluation.py` contains Precision@k, Recall@k, and Reciprocal Rank (RR) functions. To calculate meaningful metrics, create a labelled evaluation set containing question IDs and the chunk IDs known to be relevant. The project does not claim evaluation scores without such labels.

## Configuration

Environment variables are optional:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `RAG_GENERATION_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Local answer-generation model |
| `RAG_CHUNK_SIZE` | `900` | Chunk length in characters |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap in characters |
| `RAG_TOP_K` | `4` | Number of passages retrieved |

You can also change the embedding and generation model IDs in the Streamlit sidebar.

## Important limitations

- Scanned PDFs may contain images rather than selectable text. OCR is not included.
- Tables, equations, and multi-column layouts may extract incorrectly.
- Similarity scores are not proof that a passage answers the question.
- The model can still make mistakes or misrepresent retrieved context. Verify important claims against the original PDF.
- The application is a local learning project, not a production or high-impact decision system.
- Review the licenses and terms of the models and libraries you use.

## Project structure

```text
rag-document-assistant/
├── app.py
├── build_index.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   └── documents/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_processing.py
│   ├── vector_store.py
│   ├── generator.py
│   └── evaluation.py
├── storage/
│   └── .gitkeep
└── tests/
    └── test_rag_utils.py
```

## References

- PyMuPDF: https://pymupdf.readthedocs.io/
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- PyTorch installation: https://pytorch.org/get-started/locally/
