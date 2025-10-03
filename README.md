# RAG Mini (CPU) — PDFs → Questions Answered

**What:** Retrieval-Augmented Generation over your PDFs/TXTs using MiniLM embeddings + FLAN-T5 (CPU).  
**Why:** Ground answers in your data; small, reproducible stack with FastAPI, Gradio, FAISS, CI & Docker.

## Quickstart
```bash
conda activate nlp
pip install -r requirements.txt
python -m src.ingest --docs_dir docs --outdir artifacts
python -m src.rag --question "What is this project about?"
python -m src.api --host 0.0.0.0 --port 8000
# curl -s -X POST localhost:8000/ask -H "content-type: application/json" -d '{"question":"..."}'
