# Evaluate retrieval and answer grounding

Do not treat the test suite as a model-quality benchmark. It checks implementation integrity. Use this small manual evaluation before making resume or interview claims about quality.

## Build a question set before tuning

Create 12–20 questions across your five-paper collection:

- At least five narrow factual questions with known supporting pages.
- At least three method questions requiring a short explanation.
- At least two comparisons requiring evidence from two papers.
- At least two deliberately unanswerable questions.

For each, record expected paper IDs, PDF pages, and the exact supporting passages after reading them. Do not derive the ground truth from PaperIntel's generated answer. Use the same questions across comparisons of chunk size or retrieval top-k.

The bundled paper can seed questions about RAG-Sequence versus RAG-Token, retriever and generator components, datasets, and experiment limitations. Verify expected pages manually; this guide does not invent ground-truth labels.

## Record these measurements

| Measurement | Definition | Why it matters |
| --- | --- | --- |
| Evidence hit rate at k | Answerable questions with at least one genuinely supporting passage in top-k / answerable questions | Measures whether generation had relevant evidence |
| Comparison coverage | Required papers represented by supporting passages / required papers | Scores useful coverage, not just any passage per paper |
| Supported-claim rate | Factual generated claims supported by cited evidence / all factual generated claims | Detects unsupported synthesis |
| Citation correctness | Cited claims whose cited passages support them / cited claims | Separates source existence from support |
| Abstention rate on unanswerable questions | Unanswerable questions correctly declined / unanswerable questions | Measures resistance to invented answers |
| False abstention rate | Answerable questions declined despite available evidence / answerable questions | Checks excessive refusal |
| Retrieval and generation latency | Time each operation separately; note cold versus warm model/cache | Prevents mixing startup costs with routine latency |

Do not report “Recall@k” unless you have labeled the complete set of relevant units or clearly define the denominator. Evidence hit rate is an honest initial metric for a small manually labeled set.

## Suggested worksheet fields

Question ID; question text; answerable?; expected paper/page; retrieved paper/pages; top-k; retrieval model; chunk settings; Qwen model/provider; supporting hit?; generated claim count; supported claims; correct citations; abstained?; elapsed times; notes.

Save the exported retrieval and answer JSON with the worksheet so decisions can be audited. Prompts include metadata and text; a JSON character limit is not automatically a Qwen token limit. The MVP limits passage count and passage size, but does not compute the exact provider-model tokenizer budget. Keep comparisons narrow and inspect provider errors if the configured context window is small.

## After the first baseline

Identify the actual failure category before adding complexity:

- Missing candidate -> revise query or candidate retrieval.
- Wrong chunks -> revise extraction, chunking or embeddings.
- Relevant evidence ranked low -> consider evaluated hybrid retrieval or reranking.
- Relevant evidence retrieved but answer wrong -> revise context/prompt or model; check claim support.
- Broad question lacks coverage -> retrieve separately by subquestion or section under user control.

None of these changes is justified solely because it adds another framework name.
