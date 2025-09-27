# scripts/generate_numeric_qas.py
import sys, os, json, glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from src.parse_tables import cache_tables
from src.utils import append_jsonl

QA_OUT = "data/esgbench_open_source_num.jsonl"

def mk_company_from_doc(doc_name: str) -> str:
    # naive reverse of norm_doc_name: drop last two tokens (year, dtype)
    parts = doc_name.split("_")
    if len(parts) >= 3:
        return " ".join(parts[:-2]).title().replace(" And ", " & ")
    return doc_name

def mk_qa(doc_name, company, page, kpi, value, unit, raw):
    q = f"What are {company}'s {kpi} emissions in the reporting year?"
    a = f"{value} {unit}" if unit else str(value)
    return {
        "esgbench_id": None,
        "company": company,
        "doc_name": doc_name,
        "category": "Environmental",
        "question": q,
        "answer": a,
        "evidence": [{
            "evidence_text": raw[:400],
            "evidence_page_num": page,
            "evidence_doc_name": doc_name
        }],
        "units_expected": unit
    }

def main():
    for pdf in glob.glob("pdfs/*.pdf"):
        doc_name = os.path.basename(pdf)[:-4]
        company = mk_company_from_doc(doc_name)

        cache_path = cache_tables(doc_name)
        rows = json.load(open(cache_path, "r", encoding="utf-8"))

        # group by (scope, page, unit) to avoid spam
        groups = defaultdict(list)
        for r in rows:
            scope = r.get("scope")
            unit = (r.get("unit") or "").strip()
            if not scope or not unit:
                continue
            groups[(scope, r["page"], unit)].append(r)

        for (scope, page, unit), lst in groups.items():
            # pick the row with the largest numeric value (often the total)
            best = max(lst, key=lambda x: (x.get("value") or -1))
            qa = mk_qa(doc_name, company, page, scope, best["value"], unit, best["raw_text_span"])
            append_jsonl(QA_OUT, qa)
            print("[QA-num]", qa["doc_name"], qa["question"])

    print("Done numeric QAs.")

if __name__ == "__main__":
    main()
