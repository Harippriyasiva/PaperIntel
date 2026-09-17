# Architecture and decision record

## Problem and boundary

A reader can find papers but still spends time opening PDFs, finding methodological details, comparing claims, and checking where an answer came from. PaperIntel combines paper discovery with page-linked evidence retrieval and optional grounded generation.

## Two representations, two indexes

| Question | Indexed unit | Input | Lifetime |
| --- | --- | --- | --- |
| Which papers might matter? | One paper | Title + abstract | Search session |
| What evidence answers my question? | One passage | Full PDF text with page provenance | Persistent selected collection |

MiniLM produces vectors; FAISS stores and searches vectors; Qwen produces the answer. These are different jobs. Changing Qwen does not require rebuilding the embedding index. Changing the embedding model or extraction/chunk settings does.

Discovery is two-stage retrieval in the information-retrieval sense: arXiv returns lexical candidates, then dense similarity reorders them. It does not search full-text arXiv with MiniLM. Metadata filters are applied in the API query so they affect candidate retrieval rather than merely hiding already-retrieved results.

## Retain the existing UI and functions

**What:** use Streamlit, the existing color and card system, and stable discovery function names.

**Why:** the existing paper-search interaction provides a clear starting point for full-text evidence retrieval.

**How:** move network and vector operations out of the UI; provide compatibility exports in `paper_fetching.py`; replace abstract-only insight widgets with collection and evidence controls.

**Trade-off:** Streamlit reruns its script on interactions. Expensive actions belong behind buttons and caches. Static HTML uses escaped metadata. The CSS can be simplified in a later design pass.

**Interview:** “I separated application logic from the Streamlit UI so full-text retrieval can be tested without coupling it to UI reruns.”

## MiniLM and exact FAISS

**What:** normalized MiniLM embeddings and `IndexFlatIP`.

**Why:** small, CPU-friendly baseline and exact, inspectable retrieval at this scale.

**How:** normalized vectors make inner product equal cosine similarity. A score is displayed as a score, never as a percentage likelihood.

**Trade-off:** a 50-paper candidate list does not need FAISS for performance; a matrix multiplication could do the same job. FAISS gives a consistent implementation for the larger passage collection. No approximate-nearest-neighbor performance claim is justified because IndexFlatIP performs exact search. General embeddings may miss specialist distinctions. Long discovery abstracts can be truncated.

**Interview:** “I chose exact search because my corpus is small enough that approximate-index complexity would not earn its cost. I would benchmark alternatives before changing that.”

## Tokenizer-aware chunks and PDF pages

**What:** 200-token target chunks, 35-token overlap, sentence-end preference, no cross-page chunks.

**Why:** the MiniLM default input limit is 256 word pieces. Preserve evidence within that budget and retain a direct PDF page mapping.

**How:** use tokenizer character offsets to slice original normalized text; keep paper ID, 1-based PDF page, character span, token count, and deterministic chunk ID. Prefer a sentence end near the target; use a fixed limit for very long sentences. Check the model's special-token overhead before preparing a collection.

**Trade-off:** page boundaries can split concepts and overlap duplicates context. Citation simplicity is purchased at some context loss. A later improvement could use adjacent-page evidence with explicit multi-page provenance; it is not silently assumed here. Block sorting does not solve every layout. Do not describe the extractor as understanding equations or figures.

**Interview:** “Chunking is a retrieval decision: too large truncates or mixes topics, too small loses context. I kept page provenance and made the trade-off inspectable.”

## Persistence and identity

**What:** SQLite catalog, content-addressed PDFs, JSON metadata and exact FAISS indexes.

**Why:** reopening a collection should not silently substitute different papers or lose source mappings.

**How:** preserve arXiv versioned IDs; hash PDF bytes; include extraction schema and embedding-model identity in collection identity. Publish a completed directory before inserting the collection catalog entry. Failed documents are explicit manifest records, not ready entries.

**Trade-off:** this is local persistence, not a distributed service. It is not optimized for simultaneous long-running jobs. Model name is recorded, but a pinned model commit would provide stronger reproducibility than name alone. Identical preparations re-extract cached PDFs before reusing the index; page/chunk caching could reduce repeat preparation time later.

**Interview:** “The FAISS row must always map back to the same passage and PDF. I treated metadata consistency and index invalidation as system-design concerns.”

## Scoped retrieval and comparisons

**What:** restrict retrieval to the active collection, optionally selected papers. Compare 2–5 papers with up to two passages each.

**Why:** unbalanced top-k retrieval can fill the prompt with one paper and make a comparison incomplete.

**How:** exact-score the collection, filter allowed paper IDs, suppress heavily overlapping passages and apply per-paper quotas for comparison.

**Trade-off:** filtering the entire exact ranking is acceptable at this scale but not appropriate for millions of chunks. Equal evidence quotas improve coverage, not answer completeness. Questions may need multiple targeted retrievals.

**Interview:** “I distinguished relevance from coverage. A comparison needs evidence from each paper, not simply the highest scores across all papers.”

## Qwen through Hugging Face

**What:** retain the existing OpenAI-compatible Hugging Face client approach; configure the Qwen model/provider.

**Why:** keep continuity and avoid requiring a local GPU.

**How:** construct a lazy client only at generation; send question plus retrieved passage objects; set low temperature and bounded output; parse structured JSON and verify citation IDs.

**Trade-off:** needs network, provider support, account access, and available quota. This delivery cannot prove that a configured token or hard-coded provider works without a live authenticated generation. The default small Qwen may struggle with instructions or complex synthesis; structured-output failures are surfaced, not quietly rendered as success. Larger Qwen models may improve quality at higher latency/cost.

**Interview:** “I integrated a hosted instruction model for synthesis. Retrieval determines the evidence; the model does not browse or choose papers autonomously.”

## Citation trust and abstention

**What:** label evidence S1, S2, etc.; keep real page metadata outside model control.

**Why:** fabricated page numbers and invented source destinations undermine usefulness.

**How:** accept only known source IDs and reject unverified links. A substantive non-abstaining response must contain a valid citation. Show original evidence, source metadata, and a render of the PDF page.

**Trade-off:** this validates citation identifiers, not every claim. The model can omit citations on some claims or misuse a valid citation. No automatic semantic verifier is claimed. Prompts instruct abstention when evidence is insufficient, but retrieval always returns neighbors; there is no calibrated confidence threshold.

**Interview:** “Traceability and truth are different. I validate provenance structurally and evaluate support manually with a rubric rather than advertising hallucination-free answers.”

## Reference documentation checked during development

- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html)
- [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)
- [Hugging Face chat completion](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion)
- [Streamlit PDF viewer requirements](https://docs.streamlit.io/develop/api-reference/media/st.pdf)

The original `st.pdf` usage would require the additional viewer dependency. This version renders requested evidence pages with PyMuPDF instead and provides an original arXiv PDF link.
