# 🌍 ESGBench — Explainable ESG QA Benchmark

ESGBench is a small, reproducible pipeline to:

📄 Collect ESG/TCFD PDFs

📚 Build a searchable index + table cache

🤖 Auto-generate grounded ESG QA pairs with evidence

🔎 (Optional) Run a RAG baseline

📊 Evaluate predictions (EM/F1/Numeric/Recall@K)

Supports Python 3.10–3.12.

## 🚀 Installation
```bash
git clone https://github.com/<you>/esgbench
cd esgbench
```

### Choose one:
```bash
pip install -e .          # if using pyproject.toml
OR
pip install -r requirements.txt
```

Create .env from the template and add your OpenAI key:

```bash
cp .env.example .env
edit .env -> OPENAI_API_KEY=sk-xxxxxxxx
```

🔧 **Configuration System**: ESGBench now uses centralized configuration with environment variable validation. All settings can be customized via `.env` file or environment variables. See `.env.example` for available options.

## 1️⃣ Seed Documents

Edit data/docs_seed.csv (UTF-8, header required):
```bash
company,year,url,doc_type,country,industry,source
Apple Inc,2024,https://…/apple-2024-esg.pdf,ESG,US,Technology,manual
```

Then ingest (downloads PDFs to pdfs/ and logs to data/esgbench_document_information.jsonl):

```bash
python -m scripts.ingest_catalog data/docs_seed.csv
```

⚠️ If a URL 403s, it’ll be marked in the catalog and skipped.

## 2️⃣ Build Index (Chunks + Tables)

```bash
python -m scripts.build_index
```


### Outputs:

cache/chunks.json — text chunks with {doc_name, page, text}

cache/<DOC>_tables.json — parsed table rows (if found)

💡 On macOS: camelot needs Ghostscript; tabula needs Java. If table parsing fails, chunks still work.

## 3️⃣ Generate QA Pairs
python -m scripts.generate_qas_from_chunks


Appends QAs to data/esgbench_open_source.jsonl.

Each QA has:

company, doc_name, category, kpi_name, question, answer, evidence (with page number).

### Sample:

```json
{
  "company": "Apple Inc",
  "doc_name": "APPLE_INC_2024_ESG",
  "category": "Environmental",
  "kpi_name": "Scope 2 (market-based)",
  "question": "What are Apple Inc's Scope 2 (market-based) emissions in 2024?",
  "answer": "1,234,567 tCO2e",
  "evidence": [{
    "evidence_text": "… Scope 2 (market-based) were 1,234,567 tCO2e in 2024 …",
    "evidence_page_num": 39,
    "evidence_doc_name": "APPLE_INC_2024_ESG"
  }]
}
```
## Notes

✅ The generator de-dupes by (doc_name | question | answer).

🚫 Guardrails block “meta” questions (e.g., “what is item 2 in the passage”).

🔄 You can re-run safely; only new QAs are appended.

Optional numeric-only QA generator:

```bash
python -m scripts.generate_numeric_qas
```

4️⃣ (Optional) RAG Baseline → Predictions

This retrieves top-K chunks (OpenAI embeddings) and asks a model for the answer.

Set knobs (or put in .env):

```bash
export RETRIEVE_K=5
export LLM_MODEL=gpt-5-mini
export EMB_MODEL=text-embedding-3-large
```

## Run:

```bash
python -m scripts.rag_predict
```

Outputs data/esgbench_preds.jsonl:

```json
{"doc_name":"APPLE_INC_2024_ESG","question":"…","pred":"1,234,567 tCO2e","retrieved_pages":[39,40,38]}
```

5️⃣ Evaluate Predictions

```bash
python -m scripts.evaluate_esgbench data/esgbench_open_source.jsonl data/esgbench_preds.jsonl
# add -v for per-item logs
```

## Metrics:

✅ Exact Match (EM)

✍️ String F1

🔢 Numeric accuracy @±2% (unit-aware)

🔎 Retrieval Recall@K (if retrieved_pages present)

📊 Per-category accuracy

📂 Folder Layout
```bash
esgbench/
  data/                      # small seed + (generated) gold/preds
  pdfs/                      # downloaded PDFs (not committed)
  cache/                     # chunks, tables, embeddings (not committed)
  scripts/                   # CLI entry points (ingest/index/generate/eval)
  src/esgbench/              # importable library code
```

.gitignore excludes large artifacts (pdfs/, cache/, full gold/preds).

## ⚙️ Configuration

ESGBench uses a centralized configuration system with pydantic validation. Configuration is loaded from environment variables and `.env` files.

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Optional LLM Settings:**
- `LLM_MODEL` (default: gpt-4o-mini) - Model for QA generation and RAG
- `EMB_MODEL` (default: text-embedding-3-large) - Embedding model
- `API_TIMEOUT` (default: 40) - API request timeout in seconds
- `MAX_RETRIES` (default: 4) - Maximum API retry attempts

**Processing Settings:**
- `MAX_QAS_PER_DOC` (default: 16) - Maximum QA pairs per document
- `PASSAGE_CHARS` (default: 900) - Context snippet length
- `RETRIEVE_K` (default: 5) - Number of chunks to retrieve for RAG
- `INCLUDE_TABLES` (default: 1) - Whether to include table parsing

See `.env.example` for all available configuration options.

🛠 Repro Tips / Troubleshooting

❌ Nothing added to gold file → ensure cache/chunks.json exists; API key valid; check console logs.

⚠️ HTTP 403 on PDFs → leave them in seed; they’re logged as failed.

📑 Table parsing errors → still usable; QA generation also works from text.

🍎 macOS → brew install ghostscript for Camelot; install Java for Tabula.

## 📜 License & Citation

License: see LICENSE (MIT or Apache-2.0 recommended).

If you publish results using ESGBench, please cite this repo (add CITATION.cff later).

✅ One-Liner Sanity Check
### after ingest + index
```bash
head -n 5 data/docs_seed.csv
python -m scripts.generate_qas_from_chunks
python -m scripts.rag_predict
python -m scripts.evaluate_esgbench data/esgbench_open_source.jsonl data/esgbench_preds.jsonl
```

You should see non-zero EM/F1 and a growing esgbench_open_source.jsonl.
