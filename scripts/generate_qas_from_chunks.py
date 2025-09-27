# -*- coding: utf-8 -*-
"""
Generate ESG QA pairs directly from PDF passages & parsed tables using an LLM.
Run from repo root:
  python -m scripts.generate_qas_from_chunks
"""

import os, sys, re, json, glob, hashlib, random, time
from collections import defaultdict, Counter
from typing import List, Dict, Any

# Make 'src' importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import centralized configuration
from src.config import get_config, get_paths, get_llm_config, get_processing_config

# Initialize configuration
config = get_config()
paths = get_paths()
llm_config = get_llm_config()
processing_config = get_processing_config()

# Set random seed for reproducibility
random.seed(processing_config.random_seed)

# ---------------- LLM client ----------------
from openai import OpenAI, RateLimitError, APIConnectionError, APIError, BadRequestError

client = OpenAI(api_key=llm_config.openai_api_key)

# ---------------- Utilities ----------------
def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def sig(doc_name, q, a):
    return hashlib.md5(f"{doc_name}|{q.strip()}|{a.strip()}".encode()).hexdigest()

def company_from_doc(doc_name):
    parts = doc_name.split("_")
    return " ".join(parts[:-2]).title().replace(" And ", " & ") if len(parts)>=3 else doc_name

def year_from_doc(doc_name):
    parts = doc_name.split("_")
    return parts[-2] if len(parts)>=2 and parts[-2].isdigit() else "the reporting year"

NUM_UNIT_RE = re.compile(r'([-+]?[(]?\s*\d[\d,\s]*(?:\.\d+)?(?:e[+-]?\d+)?[)]?)\s*(tco2e|mtco2e|ktco2e|mwh|gj|%|usd|eur|inr)?', re.I)

def is_reasonable_numeric(ans):
    m = NUM_UNIT_RE.search(ans or "")
    return m is not None

# ---------------- Load caches ----------------
def load_chunks():
    path = paths.chunks_cache
    if not path.exists():
        print(f"[ERR] Missing {path}. Run: python scripts/build_index.py")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_table_rows_for_doc(doc_name):
    p = paths.cache_dir / f"{doc_name}_tables.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def table_row_to_prompt_snippet(row):
    headers = " | ".join(row.get("table_headers") or [])
    cells   = " | ".join([f"{k}:{v}" for k,v in (row.get("raw_cells") or {}).items() if v])
    scope   = row.get("scope")
    unit    = row.get("unit")
    val     = row.get("value")
    text    = row.get("raw_text_span") or ""
    snippet = f"HEADERS: {headers}\nCELLS: {cells}\nSCOPE: {scope}\nVALUE_UNIT: {val} {unit}\nTEXT: {text}"
    return snippet

# ---------------- Prompts ----------------
PASSAGE_PROMPT = """Create ESG/TCFD QA pairs from this sustainability report passage.

Rules (must follow all):
- Questions must target a concrete ESG KPI or governance fact (e.g., Scope 2 market-based tCO₂e, LTIR, % renewable energy, % female directors, board oversight of climate, whistleblowing, anti-corruption).
- Do NOT ask meta questions about "the passage/table/section/item/list" or positions like "item 2" / "the following".
- The answer must be factual and directly supported by the passage; copy numbers/units or short phrases verbatim.
- Prefer numeric KPIs when present; otherwise governance/strategy/risk facts.
- Output JSON list only. Each item:
  {{
    "category": "Environmental" | "Social" | "Governance" | "Strategy" | "Risk",
    "kpi_name": "short KPI label (e.g., 'Scope 2 (market-based) tCO₂e', 'LTIR', '% renewable electricity', '% female directors')",
    "question": "natural KPI question about the company/year; no references to 'passage' or 'table'",
    "answer": "verbatim value/phrase from passage",
    "evidence_quote": "verbatim supporting snippet from passage"
  }}

If nothing KPI-like is present, return [].

PASSAGE:
<<<
{passage}
>>>"""


TABLE_PROMPT = """Create ESG KPI questions from this PDF table summary (HEADERS/CELLS/TEXT).

Rules:
- Form questions about the KPI itself (e.g., Scope 2 market-based, emissions intensity, MWh, GJ, %, LTIR, % female directors).
- Do NOT mention 'table', 'row', 'cells', 'item', 'section', 'the passage' in the question.
- Answers must be verbatim values with units/symbols if present (tCO₂e, MWh, GJ, %, etc.).
- Output JSON list only. Each item has fields: category, kpi_name, question, answer, evidence_quote (copied from CELLS or TEXT).

ROW SUMMARY:
<<<
{row_summary}
>>>"""


# ---------------- LLM call with retries & logs ----------------
def ask_llm(prompt: str, tag: str) -> List[Dict[str, Any]]:
    max_retries = llm_config.max_retries
    backoff = llm_config.rate_limit_delay
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [LLM] {tag} | attempt {attempt}")
            r = client.chat.completions.create(
                model=llm_config.llm_model,
                messages=[{"role":"user","content":prompt}],
                response_format={"type":"json_object"},
                timeout=llm_config.api_timeout,
            )
            txt = r.choices[0].message.content
            try:
                j = json.loads(txt)
                if isinstance(j, list):
                    items = j
                else:
                    items = j.get("items") or j.get("data") or j.get("qas") or j.get("qa") or []
                if not isinstance(items, list):
                    items = []
            except Exception:
                items = []
            print(f"    [LLM] {tag} | ok, {len(items)} item(s)")
            return items
        except (RateLimitError, APIConnectionError, APIError, BadRequestError) as e:
            print(f"    [LLM] {tag} | error: {type(e).__name__}: {e}")
            if attempt == max_retries:
                print(f"    [LLM] {tag} | giving up after {attempt} attempts")
                return []
            sleep = backoff ** attempt + random.random()
            print(f"    [LLM] {tag} | retrying in {sleep:.1f}s …")
            time.sleep(sleep)
        except Exception as e:
            print(f"    [LLM] {tag} | unexpected error: {e}")
            return []

# ---------------- Generators ----------------
def generate_from_passages(chunks, max_per_doc, already):
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_name"]].append(c)

    kept = []
    docs = sorted(by_doc.keys())[:processing_config.head_docs]
    print(f"[passages] planning to process {len(docs)} docs (HEAD_DOCS={processing_config.head_docs})")

    for di, doc in enumerate(docs, 1):
        arr = by_doc[doc]
        company = company_from_doc(doc)
        year = year_from_doc(doc)
        random.shuffle(arr)

        print(f"[passages] Doc {di}/{len(docs)}: {doc} | trying up to {processing_config.max_passages_per_doc} passages, target {max_per_doc} QAs")
        used = 0
        seen_local = set()
        trials = 0

        for c in arr:
            if used >= max_per_doc or trials >= processing_config.max_passages_per_doc:
                break
            passage = (c.get("text") or "").strip()
            if len(passage) < 120:
                continue

            trials += 1
            short = passage[:processing_config.passage_chars]
            tag = f"{doc}:p{c['page']}:passage#{trials}"
            print(f"  [passages] {tag} | len={len(short)} chars")

            items = ask_llm(PASSAGE_PROMPT.format(passage=short), tag)
            kept_here = 0

            for it in items[:processing_config.max_qas_per_passage]:
                q = (it.get("question") or "").strip()
                a = (it.get("answer") or "").strip()
                ev = (it.get("evidence_quote") or "").strip()
                cat = (it.get("category") or "Governance").strip()
                kpi = (it.get("kpi_name") or "KPI").strip()

                if not q or not a or not ev:
                    continue
                if ev.lower() not in short.lower():
                    continue
                if any(ch.isdigit() for ch in a) and not is_reasonable_numeric(a):
                    continue

                row = {
                    "esgbench_id": None,
                    "company": company,
                    "doc_name": doc,
                    "category": cat,
                    "kpi_name": kpi,
                    "question": q.replace("{company}", company).replace("{year}", year),
                    "answer": a,
                    "evidence": [{
                        "evidence_text": ev[:400],
                        "evidence_page_num": c["page"],
                        "evidence_doc_name": doc
                    }]
                }
                signature = sig(doc, row["question"], row["answer"])
                if signature in already or signature in seen_local:
                    continue

                kept.append(row)
                seen_local.add(signature)
                already.add(signature)
                kept_here += 1
                used += 1
                print(f"    [+] passage QA added | {row['category']} | {row['kpi_name']}")
                if used >= max_per_doc:
                    break

            if kept_here == 0:
                print(f"    [~] no QA accepted from this passage")

        print(f"[passages] Doc complete: {doc} | added={used} QAs")

    return kept

def generate_from_tables(chunks, max_per_doc, already):
    docs = sorted({c["doc_name"] for c in chunks})[:processing_config.head_docs]
    print(f"[tables] planning to process {len(docs)} docs (HEAD_DOCS={processing_config.head_docs})")
    kept = []

    for di, doc in enumerate(docs, 1):
        rows = load_table_rows_for_doc(doc)
        if not rows:
            print(f"[tables] Doc {di}/{len(docs)}: {doc} | no table cache found")
            continue

        company = company_from_doc(doc)
        year = year_from_doc(doc)
        random.shuffle(rows)

        # group rows by page for variety
        rows_by_page = defaultdict(list)
        for r in rows:
            rows_by_page[r["page"]].append(r)

        print(f"[tables] Doc {di}/{len(docs)}: {doc} | pages_with_tables={len(rows_by_page)} | try up to {processing_config.max_table_pages_per_doc} pages, target {max_per_doc} QAs")
        used = 0
        seen_local = set()
        processed_pages = 0

        for page, rlist in rows_by_page.items():
            if used >= max_per_doc or processed_pages >= processing_config.max_table_pages_per_doc:
                break
            processed_pages += 1

            snippets = []
            for r in rlist[:2]:
                snippets.append(table_row_to_prompt_snippet(r))
            row_summary = "\n---\n".join(snippets)[:2500]
            tag = f"{doc}:page{page}:tables#{processed_pages}"
            print(f"  [tables] {tag} | rows_in_prompt={min(2, len(rlist))}")

            items = ask_llm(TABLE_PROMPT.format(row_summary=row_summary), tag)
            kept_here = 0

            for it in items[:processing_config.max_qas_per_passage]:
                q = (it.get("question") or "").strip()
                a = (it.get("answer") or "").strip()
                ev = (it.get("evidence_quote") or "").strip()
                cat = (it.get("category") or "Environmental").strip()
                kpi = (it.get("kpi_name") or "KPI").strip()

                if not q or not a or not ev:
                    continue
                # numeric answers from tables should look numeric if they contain digits
                if any(ch.isdigit() for ch in a) and not is_reasonable_numeric(a):
                    continue

                row = {
                    "esgbench_id": None,
                    "company": company,
                    "doc_name": doc,
                    "category": cat,
                    "kpi_name": kpi,
                    "question": q.replace("{company}", company).replace("{year}", year),
                    "answer": a,
                    "evidence": [{
                        "evidence_text": ev[:400],
                        "evidence_page_num": page,
                        "evidence_doc_name": doc
                    }]
                }
                signature = sig(doc, row["question"], row["answer"])
                if signature in already or signature in seen_local:
                    continue

                kept.append(row)
                seen_local.add(signature)
                already.add(signature)
                kept_here += 1
                used += 1
                print(f"    [+] table QA added | {row['category']} | {row['kpi_name']}")
                if used >= max_per_doc:
                    break

            if kept_here == 0:
                print(f"    [~] no QA accepted from this table page")

        print(f"[tables] Doc complete: {doc} | added={used} QAs")

    return kept

# ---------------- Entry point ----------------
def main():
    print(f"[init] MODEL={llm_config.llm_model} | INCLUDE_TABLES={processing_config.include_tables}")
    print(f"[init] Limits: MAX_QAS_PER_DOC={processing_config.max_qas_per_doc}, MAX_QAS_PER_PASSAGE={processing_config.max_qas_per_passage}, "
          f"MAX_PASSAGES_PER_DOC={processing_config.max_passages_per_doc}, MAX_TABLE_PAGES_PER_DOC={processing_config.max_table_pages_per_doc}, "
          f"PASSAGE_CHARS={processing_config.passage_chars}, HEAD_DOCS={processing_config.head_docs}")

    # load chunks from your index
    chunks = load_chunks()
    if not chunks:
        print("[exit] No chunks found; nothing to do.")
        return
    print(f"[init] Loaded {len(chunks)} chunks")

    # existing signatures to avoid duplicates across runs
    existing = set()
    for r in read_jsonl(str(paths.qa_output)):
        existing.add(sig(r["doc_name"], r["question"], r["answer"]))
    print(f"[init] Existing QA signatures: {len(existing)}")

    per_doc_passage = processing_config.max_qas_per_doc // 2
    per_doc_table = processing_config.max_qas_per_doc - per_doc_passage

    added: List[Dict[str, Any]] = []

    if processing_config.include_tables:
        print("[step] Generating from tables …")
        table_qas = generate_from_tables(chunks, per_doc_table, existing)
        for row in table_qas:
            append_jsonl(str(paths.qa_output), row)
        added.extend(table_qas)
        print(f"[step] tables added = {len(table_qas)}")

    print("[step] Generating from passages …")
    passage_qas = generate_from_passages(chunks, per_doc_passage, existing)
    for row in passage_qas:
        append_jsonl(str(paths.qa_output), row)
    added.extend(passage_qas)
    print(f"[step] passages added = {len(passage_qas)}")

    # small summary by category/kpi
    cat_ct = Counter([r["category"] for r in added])
    print("======== Summary ========")
    print("added total:", len(added))
    print("by category:", dict(cat_ct))
    print("=========================")

if __name__ == "__main__":
    main()
