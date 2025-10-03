"""
CLI RAG: retrieve with FAISS, answer with a small HF instruct model (FLAN-T5 base).
CPU-only. Prints interpreter + versions at start.
"""
import argparse, os, json, pickle, numpy as np
from typing import List, Dict
from src.utils import print_env
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

INSTRUCT_MODEL = "google/flan-t5-base"  # CPU-friendly

def load_index(outdir: str):
    idx = faiss.read_index(os.path.join(outdir, "index.faiss"))
    with open(os.path.join(outdir, "chunks.pkl"), "rb") as f:
        store = pickle.load(f)
    return idx, store["docs"], store["meta"]

def retrieve(question: str, index, docs: List[str], meta: List[Dict], embed_model: str):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(embed_model, device="cpu")
    q = enc.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, 4)
    ctxs = []
    for rank, i in enumerate(I[0]):
        if i == -1: continue
        ctxs.append({"text": docs[i], "meta": meta[i], "score": float(D[0][rank])})
    return ctxs

def build_prompt(question: str, contexts: List[Dict]) -> str:
    ctx = "\n\n".join([f"[{c['meta']['source']}#{c['meta']['chunk_id']}] {c['text']}" for c in contexts])
    prompt = (
        "Use the context to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt

def generate_answer(prompt: str, model_name=INSTRUCT_MODEL, max_new_tokens=200):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    out = mdl.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    print_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--question", required=True)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    index, docs, meta = load_index(args.artifacts)
    ctxs = retrieve(args.question, index, docs, meta, args.embed_model)
    prompt = build_prompt(args.question, ctxs)
    answer = generate_answer(prompt)

    result = {
        "question": args.question,
        "answer": answer,
        "sources": [{"source": c["meta"]["source"], "chunk_id": c["meta"]["chunk_id"], "score": c["score"]} for c in ctxs]
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
