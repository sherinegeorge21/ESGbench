import os, re, json, argparse
from collections import Counter, defaultdict

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping malformed JSON on line {i} of {path}: {e}")

def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str):
    return re.findall(r"[^\W_]+|[%/°]+", (s or "").lower())

def string_f1(gold: str, pred: str) -> float:
    gt = tokenize(gold)
    pt = tokenize(pred)
    if not gt and not pt: return 1.0
    if not gt or not pt: return 0.0
    common = Counter(gt) & Counter(pt)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pt)
    recall    = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)

NUM_RE = re.compile(r'[-+]?\d[\d,]*\.?\d*(?:e[+-]?\d+)?')
UNITS = {"tco2e","mtco2e","ktco2e","mwh","gj","%","usd","eur","inr","tco₂e"}

def numeric_match(gold: str, pred: str, rel_tol=0.02, abs_tol=1e-9) -> bool:
    def extract(x: str):
        x = (x or "").replace(",", "")
        m = NUM_RE.search(x)
        num = float(m.group(0)) if m else None
        unit = None
        xl = (x or "").lower()
        for u in UNITS:
            if u in xl:
                unit = u
                break
        return num, unit
    gnum, gunit = extract(gold)
    pnum, punit = extract(pred)
    if gnum is None or pnum is None:
        return False
    unit_ok = (gunit == punit) or (gunit is None)  # allow missing unit in pred
    if not unit_ok:
        return False
    return abs(gnum - pnum) <= max(abs_tol, rel_tol * abs(gnum))

def main():
    ap = argparse.ArgumentParser(description="Evaluate ESG-Bench predictions")
    ap.add_argument("gold_jsonl", help="Path to esgbench_open_source.jsonl")
    ap.add_argument("preds_jsonl", help="Path to predictions jsonl")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-example scores")
    args = ap.parse_args()

    print(f"[init] gold={args.gold_jsonl}")
    print(f"[init] preds={args.preds_jsonl}")

    if not os.path.exists(args.gold_jsonl):
        print("[ERR] Gold file not found."); return
    if not os.path.exists(args.preds_jsonl):
        print("[ERR] Preds file not found."); return

    gold = list(load_jsonl(args.gold_jsonl))
    preds_list = list(load_jsonl(args.preds_jsonl))
    print(f"[load] gold rows: {len(gold)}")
    print(f"[load] preds rows: {len(preds_list)}")

    # Build lookup: key = (doc_name, question) -> predicted string
    preds = {}
    drop = 0
    for r in preds_list:
        q = r.get("question")
        d = r.get("doc_name")
        if not q or not d:
            drop += 1
            continue
        # accept either "pred" or "answer" as the prediction field
        p = r.get("pred")
        if p is None:
            p = r.get("answer")
        if p is None:
            drop += 1
            continue
        preds[(d, q)] = str(p)
    if drop:
        print(f"[warn] dropped {drop} pred rows missing required fields")

    N = 0
    em = 0
    f1_sum = 0.0
    numok = 0
    cat_tot = Counter()
    cat_ok  = Counter()

    # retrieval metric (if available)
    retr_hit = 0
    retr_total = 0

    for i, g in enumerate(gold, 1):
        key = (g.get("doc_name"), g.get("question"))
        if key not in preds:
            if args.verbose: print(f"[miss] no prediction for: {key}")
            continue
        pred = preds[key]
        gold_ans = g.get("answer","")
        cat = g.get("category","NA")

        N += 1
        # EM
        if norm_text(pred) == norm_text(gold_ans):
            em += 1

        # F1
        f1 = string_f1(gold_ans, pred)
        f1_sum += f1

        # Numeric tolerance
        if any(ch.isdigit() for ch in gold_ans) and numeric_match(gold_ans, pred):
            numok += 1

        # Category bucket (credit if EM or numeric ok)
        cat_tot[cat] += 1
        if norm_text(pred) == norm_text(gold_ans) or (any(ch.isdigit() for ch in gold_ans) and numeric_match(gold_ans, pred)):
            cat_ok[cat] += 1

        # Retrieval recall@K using gold evidence page if preds included retrieved_pages
        r_pages = set()
        # find the original pred row (for retrieved_pages)
        # NOTE: if there are duplicates, the last one in preds_list wins; fine for baseline
        # we can reconstruct by scanning once:
        # (for speed, skip; this is optional)
        # but try: many pipelines store retrieved_pages
        # fallback: leave at 0 if absent
        # Better: build a dict from preds_list with key->retrieved_pages:
        # (kept simple to avoid extra memory)
        # -- optional: ignore if not present
        # (we won't error)
        # end note

        # verbose row print
        if args.verbose:
            print(f"[{i}] EM={int(norm_text(pred)==norm_text(gold_ans))} F1={f1:.2f} NUMOK={int(numeric_match(gold_ans,pred))}")
            print("  Q:", g.get("question"))
            print("  G:", gold_ans)
            print("  P:", pred)

    if N == 0:
        print("[done] 0 examples evaluated (check that questions in preds match gold).")
        return

    def pct(x): return f"{100.0*x:.2f}%"

    print("========== ESG-Bench Evaluation ==========")
    print(f"Evaluated: {N}")
    print(f"Exact Match:      {pct(em/N)}")
    print(f"String F1 (avg):  {pct(f1_sum/N)}")
    print(f"Numeric @±2%:     {pct(numok/N)}")
    print("\nCategory Accuracy (EM or Numeric OK):")
    for k in sorted(cat_tot.keys()):
        acc = (cat_ok[k]/cat_tot[k]) if cat_tot[k] else 0.0
        print(f"  {k:12s} {cat_ok[k]:4d}/{cat_tot[k]:4d}  {pct(acc)}")
    print("=========================================")

if __name__ == "__main__":
    main()
