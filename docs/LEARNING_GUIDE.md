# Understand PaperIntel by tracing one question

Do these exercises in order. The goal is to understand and debug the system, not memorize every line. Start with the example RAG paper fetched from arXiv so you can compare results with a document you can read yourself.

## 1. Draw the two searches from memory (5 minutes)

Write these two sentences in your own words:

- Discovery searches ______ using ______ and ranks candidates using ______.
- Intelligence searches ______ from papers chosen by ______, then gives evidence to ______.

Expected concepts: arXiv keyword API; titles/abstracts; MiniLM/FAISS; full-text chunks; the user; Qwen. Explain why a paper missed by arXiv candidate retrieval cannot appear through later semantic ranking.

**Read:** `paper/discovery.py`, especially `search_arxiv`, `build_faiss_index`, and `retrieve_top_papers`.

## 2. Inspect a real passage before asking Qwen (10 minutes)

Prepare the example paper, open Paper Intelligence, ask “How do RAG-Sequence and RAG-Token differ?”, and click Retrieve evidence.

For the first three results, write down the paper ID, PDF page, similarity score, and one sentence that actually helps answer the question. Open each cited PDF page. Check whether extraction preserved the text correctly.

**Important question:** If the passage is irrelevant, should you first change the generation prompt or investigate retrieval? Investigate retrieval: an LLM cannot reliably recover evidence it was never given.

**Read:** `rag/ingest.py` and `rag/retrieval.py`.

## 3. Trace one chunk all the way back (10 minutes)

Download the retrieval trace. Find one `chunk_id`. Locate that same ID in `data/collections/<collection-id>/chunks.json`. Inspect:

- `paper_id`: which paper/version it belongs to;
- `page`: 1-based PDF page number;
- `char_start` and `char_end`: its span within the normalized extracted page text;
- `token_count`: size under the chunk tokenizer;
- `text`: exact passage used for embedding and context.

Explain why a FAISS row number alone is not a useful citation. It needs a stable mapping to metadata and the source PDF.

**Read:** `rag/store.py`, especially `build` and `load`.

## 4. Separate retrieval from generation (10 minutes)

Open “Learning view: exact context sent to Qwen.” Read the system instructions and evidence objects. If a Hugging Face token is configured, generate an answer and inspect it.

Underline each factual claim. For each, identify the passage that supports it. If a citation points to a real passage but does not support the claim, mark it unsupported. Do not give credit just because a citation exists.

Explain these distinctions:

- MiniLM embeds; it does not answer the question.
- FAISS retrieves; it does not understand whether a claim is true.
- Qwen synthesizes; it does not receive the whole PDF automatically.
- A valid source identifier is not proof of factual support.

**Read:** `rag/generation.py`, especially `build_messages` and `parse_answer`.

## 5. Change a retrieval setting deliberately (15 minutes)

Use the same question with 3, 6, and 10 evidence passages. Save each trace. Which setting includes the needed method detail? Which introduces repetitive or irrelevant passages?

Then ask a more specific question. For example, ask about “RAG-Token marginalization” instead of “explain the method.” Observe whether focused wording improves retrieval.

Do not change chunk size yet. Understand top-k first. Later, a controlled experiment can compare chunk sizes of 150 and 200 tokens while keeping the same paper, questions, model, and evaluation rubric. If changing chunk settings, also change `SCHEMA_VERSION` so old indexes are not reused.

## 6. Test an unanswerable question (10 minutes)

Ask the example paper a question it does not establish, such as “What was the model's performance on a dataset introduced after this paper was written?”

The retriever will still return nearest neighbors; this does not mean an answer exists. Check whether Qwen admits missing evidence. If it invents an answer, record that as a failure in your evaluation. The current implementation does not claim guaranteed abstention.

## 7. Build a five-paper collection and compare (20 minutes)

Search an AI topic. Select five papers that actually interest you. Before preparing the collection, explain why each belongs. Do not equate a high similarity score with paper quality or influence.

Select two papers for comparison and ask a narrow shared question, such as “How does each method retrieve external evidence?” Confirm that retrieved sources contain both papers. Then compare experimental results only if their datasets and metrics are compatible; otherwise state the mismatch.

## 8. Explain a failure without guessing (10 minutes)

Use this diagnostic table:

| Symptom | First place to inspect |
| --- | --- |
| No discovery candidates | Keywords, category/year filters, arXiv availability |
| Relevant paper missing | Candidate pool, API query, metadata filters |
| First preparation fails | MiniLM download, PDF download, extraction status |
| Passage missing or scrambled | Original page versus extracted text |
| Irrelevant retrieved passages | Question, corpus scope, chunking, embedding representation |
| Qwen cannot start | Local token, configured model/provider, account access |
| Model returns invalid output | Structured response error and model capability |
| Real citation with unsupported claim | Generation grounding; manually check evidence |
| Results after changing a question | UI should require retrieval again; do not reuse stale context |

## Code-reading order

1. `README.md`: run and navigate.
2. `config.py`: fixed choices and configurable values.
3. `paper/discovery.py`: first retrieval stage.
4. `rag/ingest.py`: what actually enters the second index.
5. `rag/store.py`: identity, persistence, failures.
6. `rag/retrieval.py`: question-to-evidence mapping.
7. `rag/generation.py`: evidence-to-answer mapping.
8. `frontend.py`: how the user drives these operations.
9. `tests/test_core.py`: guarantees checked automatically and their boundaries.

## Interview rehearsal

Explain the project in 60 seconds without naming every dependency:

“PaperIntel helps readers move from finding papers to checking evidence inside them. It searches arXiv, ranks titles and abstracts with MiniLM, and lets the reader choose papers. It then extracts page-aware passages from the PDFs and retrieves relevant evidence for a question. The live app shows each passage with its source page. A configured Qwen model can use those passages to draft a cited answer, which still needs human verification.”

Use this only after you can demonstrate the actual behavior. Replace general statements with measured results once you complete the evaluation; do not invent accuracy, latency improvements, or time-saved claims.

Be ready for these questions:

- Why separate paper and chunk indexes?
- Why is inner product cosine similarity here?
- Why choose exact FAISS search for this scale?
- Why did you remove repeated titles and percentage-match labels?
- What happens if a PDF is a scan or a download fails?
- What is the difference between retrieval failure and generation failure?
- Can a valid citation still be wrong?
- What would change if the corpus grew to a million passages?
- Which PaperIntel features can you explain and demonstrate yourself?

Your first concrete task: complete exercises 1–4 and save one retrieval trace with your notes on which claims it supports.
