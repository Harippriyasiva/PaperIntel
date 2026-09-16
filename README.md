# PaperIntel

PaperIntel is a research-paper reading demo. It searches arXiv, ranks candidate
titles and abstracts with MiniLM, downloads chosen PDFs, extracts page-aware
passages, and retrieves evidence for research questions. When a Hugging Face
inference token is configured, a hosted Qwen model can compose a cited answer.

This work evolves the mentor-guided PaFet prototype. The retained ideas are
arXiv discovery, semantic ranking, and the Streamlit reader. The additions are
full-text retrieval, saved local collections, page provenance, citation checks,
and a bounded public-demo mode. It does not use LangChain, Ollama, or agents.

## Local setup

Use Python 3.12 from a terminal in this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run frontend.py
```

The first MiniLM load downloads the model. You can inspect the bundled paper
without a token. For generated answers, add your existing Hugging Face token to
a local `.env` as `HF_TOKEN=...`; `.env` is excluded from Git. Never post it in
chat or commit it. Provider availability, quotas, costs, and answer quality
still need a real check with your account.

## First demonstration

1. Select **Prepare bundled RAG paper**.
2. Ask **How do RAG-Sequence and RAG-Token differ?** in Paper Intelligence.
3. Select **Retrieve evidence** and inspect the source passages and PDF pages.
4. If a token is configured, select **Generate cited answer** and verify each
   claim against its linked passage. Citations are pointers, not proof.
5. Search another topic and prepare five papers to compare their evidence.

Normal local collections accept 5–25 papers; the bundled learning example
accepts one. Saved local collections persist in the excluded `data/` directory.

## Public demo

The public demo is live at [paperintel-haripriya.streamlit.app](https://paperintel-haripriya.streamlit.app/).
It deploys `streamlit_app.py` on Streamlit Community Cloud with Python 3.12. This
entry point keeps visitors' temporary collections separate, limits candidate
counts, and caps expensive operations. It does not provide durable user
accounts or guaranteed storage. Generation remains disabled until `HF_TOKEN`
is entered in the host's **Secrets** settings. See [deployment details](docs/DEPLOYMENT.md).

The model uses the official MiniLM ONNX file, with attention-masked pooling to
make normalized embeddings. Search uses FAISS where available and an exact
NumPy inner-product fallback elsewhere. The PDF reader uses PyMuPDF; SQLite
stores local collection metadata. The interface uses Streamlit.

## Verification and limits

Run `python -m pip install -r requirements-dev.txt` and `python -m pytest -q`.
The current Windows checks passed 26 tests, a live arXiv search, real model
loading and evidence retrieval, and processing of five real papers. The live
five-paper collection held 506 passages. On the public deployment, the bundled paper produced 123 passages and returned page-linked evidence for a RAG question. No authenticated Qwen answer has been verified. See [verification](docs/VERIFICATION.md).

arXiv limits the candidate pool; PDFs with scans, columns, tables, equations,
or figures may extract imperfectly. Citation-ID validation only checks whether
the model cites retrieved passages; it does not prove the passage supports its
claim. This is a portfolio demo, not a production multi-user research service.

Thank you to arXiv for its open-access interoperability.
