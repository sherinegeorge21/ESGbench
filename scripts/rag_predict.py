# scripts/rag_predict.py
# Generate RAG predictions for ESGBench QAs -> data/esgbench_preds.jsonl

import os, json, math, time, re, sys, hashlib
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# --- Config -------------------------------------------------------------------
QAS_PATH = os.getenv("ESGBENCH_QA", "data/esgbench_open_source.jsonl")
CHUNKS_PATH = os.getenv("ESGBENCH_CHUNKS", "cache/chunks.json")
EMB_PATH = os.getenv("ESGBENCH_EMB", "cache/chunk_embeddings.npz")   # vectors + meta
PRED_OUT = os.getenv("ESGBENCH_PREDS", "data/esgbench_preds.jsonl")

TOP_K = int(os.getenv("RETRIEVE_K", "5"))
MODEL_ANSWER = os.getenv("LLM_MODEL", "gpt-5-mini")                 # or "gpt-5"
MODEL_EMB = os.getenv("EMB_MODEL", "text-embedding-3-large")        # OpenAI embedding
PASSAGE_CHARS = int(os.getenv("PASSAGE_CHARS", "900"))              # per chunk snippet
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))     # for the answer

# --- OpenAI client ------------------------------------------------------------
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, BadRequestError
client = OpenAI()  # needs OPENAI_API_KEY

# --- I/O helpers --------------------------------------------------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- Load chunks --------------------------------------------------------------
def load_chunks(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/build_index.py")
    data = json.load(open(path, "r", encoding="utf-8"))
    # Expect each chunk has: doc_name, page, text
    return data

# --- Embeddings ---------------------------------------------------------------
def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # OpenAI embeddings are synchronous; batch by "input"
        r = client.embeddings.create(model=MODEL_EMB, input=batch)
        vecs.extend([np.array(d.embedding, dtype=np.float32) for d in r.data])
    return np.vstack(vecs)

def build_or_load_embeddings(chunks: List[dict], emb_path: str) -> Tuple[np.ndarray, List[dict]]:
    if os.path.exists(emb_path):
        z = np.load(emb_path, allow_pickle=True)
        vectors = z["vectors"]
        meta = z["meta"].tolist()
        if len(meta) == len(chunks):
            print(f"[emb] Loaded existing embeddings: {vectors.shape}")
            return vectors, meta
        else:
            print("[emb] Chunk count changed; rebuilding embeddings …")

    texts = [ (c.get("text") or "")[:2000] for c in chunks ]  # cap for cost
    print(f"[emb] Embedding {len(texts)} chunks with {MODEL_EMB} …")
    vectors = embed_texts(texts)
    meta = [{"doc_name": c["doc_name"], "page": c["page"]} for c in chunks]
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    np.savez_compressed(emb_path, vectors=vectors, meta=np.array(meta, dtype=object))
    print(f"[emb] Saved to {emb_path} -> {vectors.shape}")
    return vectors, meta

# --- Retrieval (cosine) -------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

def retrieve_for_question(
    q: str,
    doc_name: str,
    chunks: List[dict],
    vectors: np.ndarray,
    meta: List[dict],
    top_k: int = 5
) -> List[dict]:
    # restrict to this doc
    idxs = [i for i, m in enumerate(meta) if m["doc_name"] == doc_name]
    if not idxs:
        return []
    qv = embed_texts([q])
    sub = vectors[idxs]
    sims = cosine_sim(sub, qv).ravel()  # sims per chunk
    top = np.argsort(-sims)[:top_k]
    out = []
    for rank in top:
        i = idxs[rank]
        ch = chunks[i]
        out.append({
            "page": ch["page"],
            "text": (ch.get("text") or "")[:PASSAGE_CHARS]
        })
    return out

# --- Answering ---------------------------------------------------------------
ANSWER_PROMPT = """You are an ESG assistant. Answer the question using ONLY the context below.
- If a number is the answer, copy it with units/symbols as written (e.g., tCO2e, MWh, %, etc.).
- If the answer is not present, say: "Not stated in context."

Question: {q}

Context:
{ctx}
"""

def ask_llm_answer(q: str, ctx: str) -> str:
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=MODEL_ANSWER,
                messages=[{"role":"user","content":ANSWER_PROMPT.format(q=q, ctx=ctx)}],
                timeout=45
            )
            return (r.choices[0].message.content or "").strip()
        except (RateLimitError, APIConnectionError, APIError, BadRequestError) as e:
            wait = 2 ** attempt
            print(f"[llm] error {type(e).__name__} on attempt {attempt}; retrying in {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"[llm] unexpected error: {e}")
            return ""
    return ""

# --- Main --------------------------------------------------------------------
def main():
    print(f"[init] MODEL_ANSWER={MODEL_ANSWER} | MODEL_EMB={MODEL_EMB} | TOP_K={TOP_K}")
    # load data
    chunks = load_chunks(CHUNKS_PATH)
    vectors, meta = build_or_load_embeddings(chunks, EMB_PATH)

    # pre-group chunks by doc for speed (also used to check existence)
    docs_present = set([m["doc_name"] for m in meta])

    qas = list(load_jsonl(QAS_PATH))
    print(f"[data] QAs: {len(qas)} | docs in index: {len(docs_present)}")

    preds = []
    for i, row in enumerate(qas, 1):
        q = row["question"]
        doc = row["doc_name"]

        if doc not in docs_present:
            # skip if we don't have this doc indexed
            preds.append({
                "doc_name": doc,
                "question": q,
                "answer": "Not stated in context.",
                "pred": "Not stated in context.",
                "retrieved_pages": [],
            })
            continue

        hits = retrieve_for_question(q, doc, chunks, vectors, meta, TOP_K)
        ctx = "\n---\n".join([f"(page {h['page']}) {h['text']}" for h in hits]) if hits else "No context."
        pred = ask_llm_answer(q, ctx)

        preds.append({
            "doc_name": doc,
            "question": q,
            "pred": pred,
            "retrieved_pages": [h["page"] for h in hits]
        })

        if i % 25 == 0:
            print(f"[prog] answered {i}/{len(qas)}")

    write_jsonl(PRED_OUT, preds)
    print(f"[done] wrote predictions to {PRED_OUT} (rows={len(preds)})")

if __name__ == "__main__":
    main()
