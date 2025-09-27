# src/parse_tables.py
import os, json, re
import pdfplumber
from typing import List, Dict, Tuple
from rapidfuzz import fuzz

CACHE_DIR = "cache"

# tolerant number+unit regex: handles "1,234", "1 234", "1.2", "(123)", "1.2e3"
NUM_UNIT_RE = re.compile(
    r'(?P<num>\(?\s*[-+]?\d[\d,\s]*(?:\.\d+)?(?:e[+-]?\d+)?\s*\)?)\s*'
    r'(?P<unit>MtCO2e|tCO2e|ktCO2e|MWh|GJ|%)',
    re.I
)

SCOPE_HINTS = (
    ("scope 1", "Scope 1"),
    ("scope i", "Scope 1"),
    ("direct",  "Scope 1"),
    ("s1",      "Scope 1"),
    ("scope 2", "Scope 2"),
    ("scope ii","Scope 2"),
    ("s2",      "Scope 2"),
    ("scope 3", "Scope 3"),
    ("scope iii","Scope 3"),
    ("value chain","Scope 3"),
    ("s3",      "Scope 3"),
)

def _safe_to_float(num_str: str):
    """Return float or None; handles '(123)', spaces, commas, exponents."""
    if not num_str:
        return None
    s = num_str.strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "")
    s = s.replace(" ", "").replace(",", "")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None

def normalize_unit(num: str, unit: str) -> Tuple[float, str]:
    """Normalize to canonical units; return (value, unit) or (None, None)."""
    val = _safe_to_float(num)
    if val is None:
        return None, None
    u = (unit or "").lower()
    if u == "mtco2e":
        return val * 1_000_000, "tCO2e"
    if u == "ktco2e":
        return val * 1_000, "tCO2e"
    # leave %, mwh, gj, tco2e as-is
    return val, unit

def _detect_scope(cells: Dict[str,str]) -> str:
    best = None; score = 0
    for h, v in cells.items():
        s = f"{h} {v}".lower()
        for pat, lab in SCOPE_HINTS:
            sc = fuzz.partial_ratio(pat, s)
            if sc > score:
                score = sc; best = lab
    return best if score >= 80 else None

def extract_tables_pdfplumber(pdf_path: str) -> List[Dict]:
    """Return normalized rows from all tables in the PDF."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t in tables or []:
                if not t or len(t) < 2:  # need header + at least one row
                    continue
                headers = [(h or "").strip() for h in t[0]]
                for r in t[1:]:
                    # build cell dict (header->value)
                    cells = {headers[i] if i < len(headers) else f"col{i}": (r[i] or "").strip()
                             for i in range(len(r))}
                    text = " ".join(v for v in cells.values() if v)
                    if not text:
                        continue
                    m = NUM_UNIT_RE.search(text)
                    if not m:
                        continue
                    val, unit = normalize_unit(m.group("num"), m.group("unit"))
                    if val is None or not unit:
                        # skip blank/dirty numbers
                        continue
                    scope = _detect_scope(cells)
                    rows.append({
                        "page": pnum,
                        "table_headers": headers,
                        "raw_cells": cells,
                        "value": val,
                        "unit": unit,
                        "scope": scope,
                        "raw_text_span": text
                    })
    return rows

def cache_tables(doc_name: str) -> str:
    """Extract + cache table rows for a given doc_name; return cache path."""
    pdf_path = os.path.join("pdfs", f"{doc_name}.pdf")
    out_path = os.path.join(CACHE_DIR, f"{doc_name}_tables.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = extract_tables_pdfplumber(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out_path
