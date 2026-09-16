# Delivery verification

## Completed

- Read every supplied baseline code file and compared it byte-for-byte with the original ZIP contents.
- Inspected the new Paper-intel ZIP: existing `data/`, `paper/`, `rag/`, one PDF, and an empty `rag/ingest.py`.
- Preserved the original PaFet source under `original_pafet/`.
- Parsed all active application Python files successfully. The original `sample.py` syntax error remains only in the preserved, inactive baseline.
- Extracted all 19 pages of the actual supplied PDF using PyMuPDF. No page fell below the 100-character low-text warning threshold. This does not imply extraction accuracy on every page.
- Visually inspected the first original PDF page. Also inspected extracted PDF page 3: it contains the RAG-Sequence/RAG-Token explanation, but equation layout is imperfect. This is a concrete reason for the evidence-page viewer and documented limits.
- Ran `python -m pytest -q` from the project root: **22 passed**.
- Tests include real FAISS operations and real PyMuPDF extraction, using deterministic test embeddings where a model would otherwise need downloading.
- Streamlit AppTest verified token-free startup, bundled sample preparation, evidence retrieval, stale-question behavior, and selecting/preparing a five-paper collection.
- Tested the Qwen adapter with a simulated response: prompt structure, citation parsing and source-ID validation. This is not a live model test.
- Captured the installed runtime package versions in `requirements-validated.txt`. This is a version snapshot, not a fully resolved cross-platform lockfile.

## What those tests establish

- Paper metadata and PDF links are parsed and carried forward.
- Non-PDF content and redirects to unsupported hosts are rejected.
- Extracted chunk text maps back to its normalized page text and 1-based PDF page.
- Vector rows and persisted metadata survive collection reload.
- Paper-scope filtering does not return chunks from outside the chosen papers.
- Comparison evidence can include both selected papers.
- Failed documents remain explicitly failed and do not contribute chunks.
- Normal collection bounds and the one-paper learning exception are enforced.
- Unknown source IDs and non-abstaining uncited output are rejected.
- The UI can operate through its retrieval flow without an API token.

## Not verified live

The live MiniLM download and arXiv smoke-check processes did not complete: the execution environment's network approval/transport interrupted those checks. No real-model retrieval results were produced or packaged. No Hugging Face token was supplied, so no authenticated Qwen generation was attempted.

Therefore this delivery does **not** claim:

- Verified live availability of the inherited Qwen/provider identifier.
- Measured semantic retrieval accuracy or model answer quality.
- Verified provider latency, account quota, or cost.
- End-to-end live arXiv-to-Qwen success.
- A successful Windows installation; the implementation and tests ran on Linux with Python 3.12.

## Local checks to complete

1. Follow README setup on Windows.
2. Prepare the bundled sample; confirm MiniLM downloads and its index builds.
3. Ask the RAG-Sequence/RAG-Token question. Check for supporting evidence on PDF page 3; this is a known supporting page, not a promise about its retrieval rank.
4. Search arXiv for a narrow topic; confirm metadata, links, and download preparation.
5. Set `HF_TOKEN` locally; confirm the configured Qwen model/provider can serve a request, updating `QWEN_MODEL` if needed.
6. Inspect an answer claim-by-claim using its cited passages.
7. Complete the evaluation worksheet described in `docs/EVALUATION.md` before stating any quality metrics.

If a local check fails, record the exact error and the action that triggered it. Do not share `.env` contents or your token. The project already separates extraction, retrieval and generation so the failure can be diagnosed at the correct layer.
