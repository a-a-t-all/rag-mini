"""
Build FAISS index from PDFs/TXTs under docs/.
CPU-only, pinned libs. Prints interpreter + versions at start.
"""
import argparse, os, re, json, pickle, numpy as np
from typing import List, Dict, Tuple
from src.utils import print_env
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Simple chunker
def split_into_chunks(text: str, max_words: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        if not chunk: break
        chunks.append(" ".join(chunk))
        i += max(1, max_words - overlap)
    return chunks

def load_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def main():
    print_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default="docs")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_words", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = [os.path.join(args.docs_dir, f) for f in os.listdir(args.docs_dir) if f.lower().endswith((".pdf",".txt",".md"))]
    if not files:
        raise SystemExit(f"No documents found in {args.docs_dir}")

    # Read & chunk
    docs, meta = [], []
    for path in files:
        text = load_file(path)
        chunks = split_into_chunks(text, max_words=args.max_words, overlap=args.overlap)
        for j, ch in enumerate(chunks):
            docs.append(ch)
            meta.append({"source": os.path.basename(path), "chunk_id": j})

    # Embed
    model = SentenceTransformer(args.embed_model, device="cpu")
    embs = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    # FAISS index
    import faiss
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine with normalized embeddings
    index.add(embs)

    # Persist
    faiss.write_index(index, os.path.join(args.outdir, "index.faiss"))
    with open(os.path.join(args.outdir, "chunks.pkl"), "wb") as f:
        pickle.dump({"docs": docs, "meta": meta}, f)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({"embed_model": args.embed_model}, f, indent=2)

    print(f"[DONE] Indexed {len(docs)} chunks from {len(files)} files into {args.outdir}")

if __name__ == "__main__":
    main()
