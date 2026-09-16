# What changed from PaFet

## Retained and refactored

- Existing Streamlit application and visual theme/card renderer.
- arXiv API integration and title/abstract discovery representation.
- `all-MiniLM-L6-v2` embeddings.
- Normalized-vector FAISS `IndexFlatIP` ranking.
- Hosted Qwen integration through an OpenAI-compatible Hugging Face endpoint.
- Existing command-line paper search example, consolidated onto the shared client.

## Corrected

- Removed repeated titles from embedding inputs.
- Replaced percentage-match labels with similarity scores.
- Added timeouts, retry policy, API pacing, explicit ordering and metadata filters.
- Moved client creation out of module import so missing generation credentials do not block search.
- Avoided PDF downloads during ordinary display reruns.
- Escaped external paper metadata in retained HTML cards.
- Used stable paper identifiers rather than result-list positions.
- Removed abstract-only explanation from the active application; it must no longer claim to read a PDF it has not used.
- Excluded the broken FLAN-T5 experiment from runtime while preserving the original source.

## Added

- 5–25-paper selection and named collections.
- Bundled one-paper learning example using the user's actual PDF.
- Validated PDF cache and per-document preparation status.
- Page-aware extraction and tokenizer-bounded overlapping chunks.
- Persistent chunk FAISS indexes with SQLite catalog and JSON source metadata.
- Scoped passage retrieval and per-paper comparison quotas.
- Cited Qwen answer generation with structured output and source-ID checks.
- Retrieval without generation, source-page rendering, exact prompt inspection, JSON trace/answer exports.
- Offline integrity tests and interactive Streamlit flow tests.
- Windows setup instructions, architecture trade-offs, a learning guide and an evaluation plan.

## Intentionally deferred

OCR/vision, robust mathematical/table interpretation, hybrid lexical+dense retrieval, cross-encoder reranking, automatic claim-entailment verification, calibrated abstention, model fine-tuning, production authentication, distributed storage, and whole-corpus research agents. Each needs a concrete requirement or evaluation result before being added.
