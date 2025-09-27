from typing import Dict, List


def answer_from_rows(question: str, rows: List[Dict]) -> Dict:
    """
    Deterministic heuristic over structured rows (numeric/KPI).
    Replace with your LLM call to refine the selection/explanation logic.
    """
    if not rows:
        return {"status": "not_reported"}
    row = rows[0]
    return {
        "status": "ok",
        "answer_value": row.get("value"),
        "unit": row.get("unit"),
        "evidence_quote": row.get("raw_text_span", "")[:300],
        "page": row.get("page"),
        "notes": "heuristic",
    }


def answer_from_passages(question: str, passages: List[Dict]) -> Dict:
    """
    Stub to simulate an LLM RAG answer.
    Replace with your provider call and enforce citations in the response.
    """
    if not passages:
        return {"status": "not_answerable"}
    p = passages[0]
    return {
        "status": "ok",
        "answer_text": "Yes",
        "evidence_quote": p["text"][:300],
        "page": p["page"],
        "notes": "heuristic",
    }
