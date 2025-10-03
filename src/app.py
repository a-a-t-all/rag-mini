"""
Gradio wrapper for RAG. CPU-only. Prints env on startup.
"""
import argparse, os, json, pickle
from src.utils import print_env
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    print_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--gen_model", default="google/flan-t5-base")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    index = faiss.read_index(os.path.join(args.artifacts, "index.faiss"))
    import pickle
    with open(os.path.join(args.artifacts, "chunks.pkl"), "rb") as f:
        store = pickle.load(f)
    docs, meta = store["docs"], store["meta"]

    embedder = SentenceTransformer(args.embed_model, device="cpu")
    tok = AutoTokenizer.from_pretrained(args.gen_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model)

    def ask(q):
        qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, 4)
        ctxs = []
        for rank, i in enumerate(I[0]):
            if i == -1: continue
            ctxs.append((docs[i], meta[i], float(D[0][rank])))
        ctx_txt = "\n\n".join([f"[{m['source']}#{m['chunk_id']}] {t}" for (t,m,_) in ctxs])
        prompt = f"Use the context to answer. If unknown, say so.\n\nContext:\n{ctx_txt}\n\nQuestion: {q}\nAnswer:"
        enc = tok(prompt, return_tensors="pt", truncation=True)
        out = mdl.generate(**enc, max_new_tokens=200)
        ans = tok.decode(out[0], skip_special_tokens=True)
        sources = "\n".join([f"{m['source']} (chunk {m['chunk_id']}, score {s:.3f})" for (_,m,s) in ctxs])
        return ans, sources

    demo = gr.Interface(
        fn=ask,
        inputs=gr.Textbox(lines=2, label="Question"),
        outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
        title="RAG Mini (CPU)",
        description="Retrieval-augmented QA over your PDFs/TXTs."
    )
    demo.launch(server_name=args.host, server_port=args.port)

if __name__ == "__main__":
    main()
