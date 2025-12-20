# RAG Demo (fastembed + OpenAI)

Small demo repository showing a minimal Retrieval-Augmented Generation (RAG) workflow:
- Use fastembed to produce embeddings for documents and queries.
- Use cosine similarity to rank documents relevant to a query.
- (OpenAI client is present for downstream LLM calls if you extend the demo.)

## Requirements
# RAG Demo (fastembed + OpenAI)

Small demo repository demonstrating a few simple Retrieval-Augmented Generation (RAG) approaches using local embeddings, LlamaIndex, and an optional vector store.

Compared techniques
- FastEmbed + manual cosine-ranking: produce embeddings locally with `fastembed` and rank documents by cosine similarity (simple, local).
- LlamaIndex (auto-embeds via OpenAI): let LlamaIndex call OpenAI to produce embeddings and perform retrieval.
- LlamaIndex configured to use FastEmbed: use local `fastembed` for embeddings while using LlamaIndex for indexing and querying (no OpenAI embed calls).
- LlamaIndex + Qdrant vector store: persist embeddings in Qdrant and use LlamaIndex for retrieval and OpenAI for generation.

Prerequisites
- Python 3.8+
- Dependencies are listed in `pyproject.toml`.

Install with pip:

```bash
pip install fastembed scikit-learn numpy openai
```

Or sync dependencies with `uv` (optional, `uv.lock` is included):

```bash
uv sync
```

OpenAI API key
- Flows that call OpenAI (auto-embeds or generation) require an API key. Export it before running examples.

macOS / Linux (bash / zsh):

```bash
export OPENAI_API_KEY="sk-..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Windows CMD (session only):

```cmd
set OPENAI_API_KEY=sk-...
```

Usage
- Run the main demo example:

```bash
uv run streamlit run main.py
```

Files of interest
- `main.py`: minimal demo runner that embeds and ranks.
- `pyproject.toml`: dependency manifest.
- `uv.lock`: lockfile for `uv`-based dependency syncing (optional).

Notes
- This README highlights the four techniques being compared and points to example code. Modify `main.py` or the notebook to try different embedding models, vector stores, or LLM backends.
