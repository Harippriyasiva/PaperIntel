# PaperIntel

**Explore research papers. Ask focused questions. Trace the evidence back to the page.**

PaperIntel is a Streamlit application for discovering arXiv papers and examining
the evidence inside selected PDFs. It ranks paper titles and abstracts with
MiniLM, builds a searchable collection of page-aware passages, and shows the
source page for every retrieved result. When a Hugging Face token is configured,
Qwen can draft a cited answer from those passages for the reader to verify.

**[Open PaperIntel](https://paperintel-haripriya.streamlit.app/)**

## What you can do

- Search arXiv and choose papers for a reading collection.
- Read the original PDFs and retrieve passages relevant to a question.
- Follow each result to its PDF page and download a retrieval trace.
- Compare evidence across selected papers.
- Generate a cited answer when a compatible Qwen provider is configured.

## Try it

1. In the live app, select **Prepare example RAG paper**. PaperIntel fetches it from arXiv on demand.
2. Open **Paper Intelligence** and ask **How do RAG-Sequence and RAG-Token differ?**
3. Select **Retrieve evidence**. Open the page-linked passages to inspect the source.

The public app was verified with this workflow: the paper produced 19 extracted
pages and 123 searchable passages, and the question returned six page-linked
results. Generated answers are currently unavailable because the hosted app has
no `HF_TOKEN` configured.

## How it works

| Step | What PaperIntel does |
| --- | --- |
| Discover | Searches arXiv and ranks title/abstract candidates with MiniLM. |
| Prepare | Extracts PDF text into overlapping passages that retain paper and page metadata. |
| Retrieve | Searches those passages with FAISS, or an exact NumPy fallback, and displays source pages. |
| Answer (optional) | Sends retrieved passages to a configured Qwen provider and checks cited source IDs. |

The public app keeps each visitor's collection temporary and separate. Local
collections can be saved and reopened. See [deployment and operational details](docs/DEPLOYMENT.md)
and the [verification record](docs/VERIFICATION.md).

## Run locally

Use Python 3.12 from this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run frontend.py
```

Retrieval works without an inference token. To try generated answers locally,
put your Hugging Face token in an untracked `.env` as `HF_TOKEN=...`. Set a
compatible `QWEN_MODEL` if your provider requires one. Keep credentials out of
Git; see [`.env.example`](.env.example) for the expected settings.

Run the checks with `python -m pip install -r requirements-dev.txt` and
`python -m pytest -q`. The current release passed 26 automated tests on
Windows and processed five real papers locally into 506 passages. The public
sample workflow was also verified. Answer quality and citation support still
need a live Qwen check and evaluation; [the evaluation plan](docs/EVALUATION.md)
describes that work.

The example PDF is fetched from [the original arXiv record](https://arxiv.org/abs/2005.11401v4)
when requested; no third-party paper PDF is included in this repository.

PaperIntel uses PyMuPDF for PDF text, SQLite for local collection metadata,
MiniLM ONNX for embeddings, and Streamlit for the interface. Thanks to arXiv
for making open-access paper discovery possible.
