import json
import os
from typing import Dict

from .llm_iface import answer_from_passages, answer_from_rows
from .normalizers import is_numeric_question
from .parse_tables import cache_tables
from .retriever import HybridRetriever, pdf_to_chunks


def answer_question(doc_name: str, question: str) -> Dict:
    if is_numeric_question(question):
        cache_path = cache_tables(doc_name)
        rows = json.load(open(cache_path, "r", encoding="utf-8"))
        return answer_from_rows(question, rows)

    chunks = pdf_to_chunks(doc_name)
    retr = HybridRetriever(chunks)
    topk = retr.search(question, topk=6)
    return answer_from_passages(question, topk)
