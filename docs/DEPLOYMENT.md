# Deploy the portfolio demo

Deploy `streamlit_app.py` on Streamlit Community Cloud with Python 3.12.
Use the requirements.txt in the repository root. The public entry point enables
session-isolated temporary storage. Local persistent mode remains `frontend.py`.

1. Put only this release source in a personal GitHub repository. Do not use
   similarly named repositories belonging to other people or lab projects.
2. Sign in at https://share.streamlit.io/ and create an app from that repository.
3. Select `main`, entry point `streamlit_app.py`, and Python 3.12.
4. Add HF_TOKEN and QWEN_MODEL in the host's Secrets settings to enable answers.
   The selected Hugging Face model/provider must be available to your account.
5. Deploy. Open the resulting URL in a new browser session.
6. Search, prepare five papers, retrieve evidence, generate one answer, and inspect citations.

Never upload .env, .venv, local data, personal backups, or secrets.toml.
The release ZIP excludes those files. A GitHub source URL is not an app deployment.

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

This is a small portfolio demo, not a production multi-user research service.

## Runtime

MiniLM uses its official ONNX model with attention-masked mean pooling and normalized
384-dimensional vectors. This removes the PyTorch dependency from the demo.
FAISS is used where available; exact NumPy inner-product search is the fallback.
Older FAISS-only collections may need preparing again on Windows. Original data is retained.

Official hosting documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
