PAII Systems
=================

Purpose
-------
PAII Systems is a lightweight, local-first vector search tool that ingests
text (including PDFs), embeds content, stores embeddings in FAISS, and provides
a simple CLI and programmatic API for searching similar passages. The project
supports both local (sentence-transformers) and remote (OpenAI) embeddings via
an abstracted provider interface.

What this should do
--------------------
- Ingest plain text and PDF documents, split them into stable chunks with
  metadata.
- Produce embeddings via a configurable provider (local SentenceTransformer or
  OpenAI Embeddings).
- Persist embeddings in a FAISS index and store associated metadata in a
  JSONL metadata file (one entry per chunk).
- Provide a CLI and programmatic API for searching the index and returning best
  matching chunks with provenance and stable scores.

High-level logic flow
---------------------
1. Ingest: read source (plain text or PDF) -> clean -> chunk (size + overlap).
2. Embed: batch chunks and convert to np.float32 vectors using the
   configured embedding provider.
3. Persist: add vectors to FAISS (validate shape/dtype) and append chunk
   metadata to `data/text_data.jsonl` atomically.
4. Search: embed query -> run FAISS search -> map indices to metadata -> score
   & return results.

Inference Providers (modular)
-----------------------------
This project keeps inference strictly modular so you can swap LLM backends
without changing the retrieval or storage layers. Three provider types are
supported (and extensible):

- **OpenAI**: Use the OpenAI API for generation and chat-style completions.
- **Deepseek**: Call a Deepseek-compatible JSON API (adapter included).
- **Local**: Run a small local model (e.g. `distilgpt2` or other HF causal LMs)
  via the `transformers` pipeline for offline / privacy-preserving inference.

You can toggle providers via environment variables or CLI flags (examples:
`--llm-provider openai|deepseek|local` and `--provider local|openai` for
embeddings). The `paii/llm.py` adapter provides a single `LLMModel`
interface so `PAIISystem` can call `generate(prompt, ...)` regardless of
the underlying provider.

Recommended repository layout
----------------------------
- `README.md` — this file.
- `pyproject.toml` / `requirements.txt` — dependency and packaging information.
- `LICENSE` and `.gitignore`.
- `src/paii/` — package source code
  - `__init__.py` — package exports and version
  - `app.py` — `PAIISystem` orchestration (public API: `add`, `ingest_pdf`,
    `search`).
  - `cli.py` — CLI entrypoint (prefer `typer` or `argparse`), parses flags and
    delegates to `app`.
  - `embeddings.py` — `EmbeddingModel` interface and provider implementations:
    `SentenceTransformerEmbedding` and `OpenAIEmbedding`, plus a provider factory.
  - `db.py` — `FaissStore` wrapper: index creation/management, atomic save/load,
    JSONL metadata management, `add()` and `search()` methods returning structured
    results.
  - `pdf.py` — PDF extraction and chunking logic (configurable chunk size and
    overlap) returning text + metadata.
  - `utils.py` — helpers: chunking strategies, cleaning, `ensure_2d_float32`,
    scoring conversions and small I/O helpers.
  - `config.py` — default constants and environment overrides (e.g.
    `OPENAI_API_KEY`, index paths).
- `data/` (gitignored): runtime artifacts
  - `data/faiss_index.bin` — FAISS binary index
  - `data/text_data.jsonl` — line-delimited JSON metadata entries
- `tests/` — unit tests (embeddings, db, pdf chunking).
- `examples/` — small example scripts or notebooks.

Per-file responsibilities (brief)
--------------------------------
- `src/paii/app.py`: Compose components, validate inputs, and expose the
  programmatic API used by the CLI and tests. Handles high-level operations
  (ingest, add, search) and error reporting.
- `src/paii/cli.py`: Minimal command-line interface that maps flags to the
  `app` API and handles environment variables for provider configuration.
- `src/paii/embeddings.py`: Abstraction for embeddings providers. Each
  provider must return a NumPy `np.float32` array with shape `(N, dim)`.
  Implement batching, retries, and OpenAI key validation here.
- `src/paii/db.py`: Encapsulate FAISS logic and metadata mapping. Ensure
  atomic writes, validate vector shapes, and provide clear `Result` objects
  containing `text`, `score`, and `metadata`.
- `src/paii/pdf.py`: Extract text with `PyMuPDF` (`fitz`), chunk into
  paragraphs/sentences with overlap, and return items ready for embedding and
  indexing.
- `src/paii/utils.py`: Reusable helpers for chunking, cleaning, vector
  normalization, and scoring conversions (distance -> similarity).
- `src/paii/config.py`: Centralized defaults and environment-driven
  overrides for model names, index paths, and other runtime configuration.

Data format recommendations
---------------------------
- Use `data/text_data.jsonl` with one JSON object per chunk, for example:

  {
    "id": 123,
    "text": "...",
    "source": "file.pdf",
    "chunk_id": 3,
    "created_at": "2025-12-06T12:34:56Z"
  }

- Store the FAISS index in `data/faiss_index.bin`. Always write temp files and
  perform atomic replace to avoid corruption.

Quickstart (developer)
----------------------
Install dependencies and create a venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ingest a PDF using local embeddings:

```bash
python -m src.paii.cli --pdf examples/sample.pdf --provider local
```

Query the index:

```bash
python -m src.paii.cli --query "What is the main idea of chapter 2?" --top-k 5
```

Migration notes (from current project)
-------------------------------------
- Replace the newline-joined `text_data.txt` with `text_data.jsonl` to preserve
  metadata and enable robust deduplication and provenance tracking.
- Consolidate duplicated code in `paii.py` and `paii2.py` into the
  package structure under `src/paii/`.
- Validate and enforce embedding shapes/dtypes before adding to FAISS.

Next steps I can take
---------------------
- Scaffold the `src/paii/` package and implement core modules
  (`embeddings.py`, `db.py`, `pdf.py`, `cli.py`).
- Add unit tests and a minimal `pyproject.toml` / `requirements.txt`.

Tell me which next step you want and I will scaffold it.
This method fully runs locally using FAISS for vector search and sentence-transformers for text embeddings. No external API calls (e.g., OpenAI) are needed.

Key Components
Text Embedding Model

Uses sentence-transformers (all-MiniLM-L6-v2) to convert text into vector embeddings.
FAISS Vector Store

A flat L2 index (IndexFlatL2) is created to store and search for embeddings.
This index enables fast similarity search.
Core Functions

process_input(): Handles CLI input.
embed_text(text): Converts text into an embedding.
add_to_db(text): Converts text to an embedding and stores it in FAISS.
search_db(query, top_k=3): Searches for the most similar stored embeddings in FAISS.
How It Works
User provides input text.
Embedding is generated using sentence-transformers.
Text embedding is stored in FAISS (add_to_db).
User queries the system, and FAISS searches for the most similar stored text (search_db).
Pros & Cons
✅ Fully local (No API calls, No internet required).
✅ Cost-free (No OpenAI API costs).
✅ Fast (FAISS optimizes nearest neighbor search).

⚠️ Embedding quality is lower compared to OpenAI’s models.
⚠️ Limited to pre-trained transformer model (all-MiniLM-L6-v2).
⚠️ No text generation (Unlike OpenAI’s completion models).

When to Use This?
If you want full privacy and zero API costs.
If you only need fast similarity search and not AI-generated responses.
If you plan to scale locally without cloud dependencies.


i am updating this project to use both local and openai embeddings