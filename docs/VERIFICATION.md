# Verification record

## Confirmed

- The release passed 26 automated tests on Windows with Python 3.12.
- Local checks loaded the real MiniLM ONNX model, searched arXiv, retrieved evidence, and prepared five real papers into 506 passages.
- The public Streamlit app started with Python 3.12 and installed dependencies from `requirements.txt`.
- In the hosted app, the bundled RAG paper produced 19 extracted pages and 123 searchable passages.
- The hosted question “How do RAG-Sequence and RAG-Token differ?” returned six page-linked evidence passages; the top three pointed to PDF page 3.
- The hosted interface shows that generation is unavailable without `HF_TOKEN`.

## Still to verify

- A real authenticated Qwen response, including whether the selected provider and model are available to the account.
- Claim-by-claim support of a generated answer against its cited passages.
- Hosted five-paper search and preparation under public traffic; the five-paper flow was checked locally.
- Retrieval quality across a larger evaluation set. The [evaluation plan](EVALUATION.md) defines how to measure it.

Retrieved passages and valid citation IDs make an answer inspectable; they do not by themselves prove that every claim is correct.
