import re
from typing import Tuple


def normalize_num_unit(s: str):
    m = re.search(r"([\d\.,]+)\s*(MtCO2e|tCO2e|ktCO2e|MWh|GJ|%)?", s, re.I)
    if not m:
        return None, None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "mtco2e":
        val *= 1_000_000
        unit = "tco2e"
    if unit == "ktco2e":
        val *= 1_000
        unit = "tco2e"
    return val, unit


def exact_or_tolerance(pred: str, gold: str, tol_pct: float = 0.02) -> bool:
    pv, pu = normalize_num_unit(pred)
    gv, gu = normalize_num_unit(gold)
    if pv is None or gv is None:
        return False
    if pu and gu and pu != gu:
        return False
    if gv == 0:
        return abs(pv - gv) < 1e-9
    return abs(pv - gv) / abs(gv) <= tol_pct


def faithfulness_contains(quote: str, gold_span: str) -> bool:
    if not quote or not gold_span:
        return False
    return gold_span.lower().strip() in quote.lower()


def classification_prf(pred_labels, gold_labels) -> Tuple[float, float, float]:
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, average="macro", zero_division=0
    )
    return p, r, f
