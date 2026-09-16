# Deployment and operations

PaperIntel is live at [paperintel-haripriya.streamlit.app](https://paperintel-haripriya.streamlit.app/).
Streamlit Community Cloud deploys `streamlit_app.py` from the `main` branch of
[Harippriyasiva/PaperIntel](https://github.com/Harippriyasiva/PaperIntel) with
Python 3.12 and the repository's `requirements.txt`. The public entry point
uses temporary, session-isolated storage. Local persistent mode uses `frontend.py`.

The hosted app was checked with the bundled RAG paper: it extracted 19 pages,
indexed 123 passages, and returned page-linked evidence for a question about
RAG-Sequence and RAG-Token. Public arXiv search and five-paper preparation were
tested locally. Answer generation still needs a configured Hugging Face token
and a live provider check.

To enable generated answers, configure `HF_TOKEN` and, if needed, `QWEN_MODEL`
in the app's Streamlit Secrets settings. Test the selected model with your
Hugging Face account before treating generated answers as verified. Keep
tokens in Secrets; `.env`, `.venv`, local data, and `secrets.toml` are excluded
from the repository.

## Demo behavior and limits

- Each browser session gets a separate temporary directory. These are not durable accounts.
- Sessions cannot open each other's catalog through the app interface.
- Downloads let readers save evidence and explanations before leaving.
- Public collection preparation accepts five papers, plus the single learning sample.
- Public search is bounded to 50 candidate papers; select 25 or 50 in the UI.
- Per-process hourly limits: 20 answer attempts, 12 preparation attempts, 60 previews, 60 searches.
- Limits reset with the server process; they are not a billing cap or distributed rate limiter.
- PDF downloads are limited to 40 MB; extraction supports at most 150 pages per PDF.
- A provider token enables paid/quotad inference. Configure provider-side spending restrictions.
- No user accounts, OCR, durable cloud collections, abuse-resistant distributed quotas, or availability SLA.

## Runtime

MiniLM uses its official ONNX model with attention-masked mean pooling and normalized
384-dimensional vectors. This removes the PyTorch dependency from the demo.
FAISS is used where available; exact NumPy inner-product search is the fallback.
Older FAISS-only collections may need preparing again on Windows. Original data is retained.

Official hosting documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
