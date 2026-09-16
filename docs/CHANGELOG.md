# PaperIntel release notes

## 0.4.0 — public demo

- Published the Streamlit app at [paperintel-haripriya.streamlit.app](https://paperintel-haripriya.streamlit.app/).
- Added session-isolated temporary collections and bounded public-demo operations.
- Verified hosted preparation of the bundled paper (19 pages, 123 passages) and page-linked retrieval.
- Kept generated answers unavailable until a Hugging Face token is configured.

## Research workflow

- Search arXiv and rank paper titles and abstracts with MiniLM embeddings.
- Select papers, extract PDF text by page, and build a searchable passage collection.
- Retrieve passages scoped to selected papers, inspect source pages, and export a retrieval trace.
- Support cited answer generation when a compatible Qwen provider is configured.

## Reliability and review

- Added PDF validation, download limits, preparation status, and stable paper identifiers.
- Added source-ID checks for generated citations and clear display of similarity scores.
- Added automated tests and an evaluation worksheet for checking evidence quality.
