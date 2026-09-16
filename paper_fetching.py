"""Compatibility imports for PaFet's discovery functions.

The abstract-only generate_explanation function was deliberately replaced by
rag.generation.generate(question, retrieved_passages). See docs/DECISIONS.md.
"""
from paper.discovery import search_arxiv, load_model, build_faiss_index, retrieve_top_papers
from paper.documents import download_pdf_bytes


def download_pdf(pdf_url, title):
    filename = ''.join(c for c in title if c.isalnum() or c in ' _-').strip()[:140] or 'paper'
    return download_pdf_bytes(pdf_url), filename + '.pdf'
