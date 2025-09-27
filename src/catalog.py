import os, requests, urllib.parse, re
from typing import Optional
from pydantic import BaseModel, Field

from .utils import append_jsonl, norm_doc_name, sha256_file

CATALOG = "data/esgbench_document_information.jsonl"
PDF_DIR = "pdfs"

class DocInfo(BaseModel):
    doc_name: str
    doc_type: str = Field(default="ESG")
    doc_period: int
    doc_link: str
    company: str
    industry: Optional[str] = None
    country: Optional[str] = None
    source: Optional[str] = None
    status: str = "pending"          # "downloaded" | "error:download"
    sha256: Optional[str] = None
    error: Optional[str] = None      # capture reason for failures

def _origin(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.netloc}"

def _sanitize_dtype(dtype: str) -> str:
    # avoid slashes/spaces in filenames
    return re.sub(r"[^A-Za-z0-9]+", "_", dtype or "ESG")

def download_pdf(url: str, out_path: str, timeout: int = 90):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # try vanilla
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
    except Exception:
        # retry with browser-like headers + referer (helps for 403s)
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/131.0.0.0 Safari/537.36"),
            "Accept": "application/pdf,*/*;q=0.9",
            "Referer": _origin(url),
            "Connection": "close",
        }
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def add_doc(company: str, year: int, url: str, dtype: str = "ESG", **kw):
    dtype = _sanitize_dtype(dtype)
    doc_name = norm_doc_name(company, year, dtype)
    out_path = os.path.join(PDF_DIR, f"{doc_name}.pdf")

    digest: Optional[str] = None
    status = "downloaded"
    err_msg: Optional[str] = None

    try:
        download_pdf(url, out_path)
        digest = sha256_file(out_path)
    except Exception as e:
        status = "error:download"
        err_msg = str(e)

    doc_info = DocInfo(
        doc_name=doc_name,
        doc_type=dtype,
        doc_period=year,
        doc_link=url,
        company=company,
        status=status,
        sha256=digest,
        error=err_msg,
        **kw
    )

    append_jsonl(CATALOG, doc_info.model_dump())
    return doc_name
