"""
FastAPI server for RAG (/ask). CPU-only. Prints env on startup.
"""
import argparse, os, json, pickle
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import print_env
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class AskReq(BaseModel):
    question: str

def build_app(artifacts: str, embed_model: str, gen_model: str):
    print_env()
    app = FastAPI(title="RAG Mini (CPU)")
    # Load index
    index = faiss.read_index(os.path.join(artifacts, "index.faiss"))
    with open(os.path.join(artifacts, "chunks.pkl"), "rb") as f:
        store = pickle.load(f)
    docs, meta = store["docs"], store["meta"]
    # Load models
    embedder = SentenceTransformer(embed_model, device="cpu")
    tok = AutoTokenizer.from_pretrained(gen_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_model)

    @app.post("/ask")
    def ask(req: AskReq):
        q = embedder.encode([req.question], normalize_embeddings=True).astype("float32")
        D, I = index.search(q, 4)
        ctxs = []
        for rank, i in enumerate(I[0]):
            if i == -1: continue
            ctxs.append((docs[i], meta[i], float(D[0][rank])))

        ctx_txt = "\n\n".join([f"[{m['source']}#{m['chunk_id']}] {t}" for (t,m,_) in ctxs])
        prompt = (
            "Use the context to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{ctx_txt}\n\nQuestion: {req.question}\nAnswer:"
        )
        enc = tok(prompt, return_tensors="pt", truncation=True)
        out = mdl.generate(**enc, max_new_tokens=200)
        ans = tok.decode(out[0], skip_special_tokens=True)
        return {"answer": ans,
                "sources": [{"source": m["source"], "chunk_id": m["chunk_id"], "score": s} for (_,m,s) in ctxs]}

    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--gen_model", default="google/flan-t5-base")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = build_app(args.artifacts, args.embed_model, args.gen_model)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
