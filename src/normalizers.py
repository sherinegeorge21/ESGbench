import re

NUMERIC_HINTS = [
    r"\b(scope\s*[123]|ghg|emission|co2e|ltir|injury|renewable|energy|mwh|gj|%)\b"
]


def is_numeric_question(q: str) -> bool:
    ql = q.lower()
    return any(re.search(p, ql) for p in NUMERIC_HINTS)
