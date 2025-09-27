import json
import os
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(self, chunks: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.ids = [c["id"] for c in chunks]
        self.tok = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tok)
        self.model = SentenceTransformer(model_name)
        X = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.X = X

    def search(self, query: str, topk: int = 6):
        xq = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(xq.astype(np.float32), topk)
        dense_hits = [(self.ids[i], float(D[0][j])) for j, i in enumerate(I[0])]
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_hits = sorted(
            [(self.ids[i], float(bm25_scores[i])) for i in range(len(self.ids))],
            key=lambda x: -x[1],
        )[:topk]
        scores = {}
        for i, s in dense_hits + bm25_hits:
            scores[i] = scores.get(i, 0.0) + s
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:topk]
        return [next(c for c in self.chunks if c["id"] == cid) for cid, _ in ranked]


def pdf_to_chunks(doc_name: str, max_chars: int = 1200, overlap: int = 200):
    txt_path = os.path.join("cache", f"{doc_name}_text.json")
    if not os.path.exists(txt_path):
        import pdfplumber

        arr = []
        with pdfplumber.open(os.path.join("pdfs", f"{doc_name}.pdf")) as pdf:
            for pnum, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                arr.append({"page": pnum, "text": t})
        os.makedirs("cache", exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            json.dump(arr, f)
    with open(txt_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    chunks = []
    cid = 0
    for p in pages:
        t = p["text"] or ""
        i = 0
        while i < len(t):
            chunk = t[i : i + max_chars]
            if chunk.strip():
                chunks.append(
                    {
                        "id": f"{doc_name}:{p['page']}:{cid}",
                        "doc_name": doc_name,
                        "page": p["page"],
                        "text": chunk,
                    }
                )
                cid += 1
            i += max(1, max_chars - overlap)
    return chunks
