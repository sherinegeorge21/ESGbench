# scripts/ingest_catalog.py
import os, sys, csv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # repo root
from src.catalog import add_doc

"""
Usage:
  python scripts/ingest_catalog.py data/docs_seed.csv

Reads a CSV with columns:
company,year,url,doc_type,industry,country,source
Downloads each PDF into pdfs/ and appends a JSON line into
data/esgbench_document_information.jsonl (resume-safe).
"""

def main(csv_path: str):
    added = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            company = row["company"].strip()
            year = int(row["year"])
            url = row["url"].strip()
            dtype = (row.get("doc_type") or "ESG").strip()
            kw = {}
            for k in ("industry","country","source"):
                if row.get(k):
                    kw[k] = row[k].strip()
            doc_name = add_doc(company, year, url, dtype=dtype, **kw)
            print(f"[OK] {doc_name}")
            added += 1
    print(f"Done. Added {added} docs.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_catalog.py data/docs_seed.csv")
        sys.exit(1)
    main(sys.argv[1])
