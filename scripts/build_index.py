import os, json, pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, hashlib, argparse
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # disable TF/Keras path
os.environ["TRANSFORMERS_NO_FLAX"] = "1" # optional, keep Flax off too
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet tokenizer warnings


PDF_DIR = "pdfs"
CACHE_DIR = "cache"

def text_chunks_for_pdf(pdf_path, doc_name, max_chars=1200, overlap=200):
    arr=[]
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            i=0
            while i < len(t):
                chunk = t[i:i+max_chars]
                if chunk.strip():
                    cid = hashlib.md5(f"{doc_name}:{pnum}:{i}".encode()).hexdigest()[:12]
                    arr.append({"id":f"{doc_name}:{pnum}:{cid}", "doc_name":doc_name, "page":pnum, "text":chunk})
                i += max(1, max_chars-overlap)
    return arr

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks=[]
    for fn in os.listdir(PDF_DIR):
        if not fn.lower().endswith(".pdf"): 
            continue
        doc_name = fn[:-4]
        pdf_path = os.path.join(PDF_DIR, fn)
        chunks = text_chunks_for_pdf(pdf_path, doc_name)
        all_chunks.extend(chunks)
        print(f"[chunks] {doc_name}: {len(chunks)}")

    texts = [c["text"] for c in all_chunks]
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)

    with open(os.path.join(CACHE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    faiss.write_index(index, os.path.join(CACHE_DIR, "faiss.index"))
    print(f"Indexed {len(all_chunks)} chunks.")

if __name__ == "__main__":
    main()
