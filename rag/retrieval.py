"""Exact passage retrieval, collection isolation, and balanced paper comparisons."""
import numpy as np
from paper.discovery import encode


def retrieve(question, collection, model, paper_ids=None, k=6, compare=False):
    from paper import vector_index as faiss
    if not question.strip():
        raise ValueError('Enter a question.')
    ready = {p['paper_id'] for p in collection['manifest']['papers'] if p['status'] == 'ready'}
    scope = set(paper_ids) if paper_ids is not None else ready
    if not scope or not scope <= ready:
        raise ValueError('Choose papers that are ready in the active collection.')
    if compare and not 2 <= len(scope) <= 5:
        raise ValueError('Compare 2–5 papers at once so each receives enough evidence.')
    index, chunks = collection['index'], collection['chunks']
    vector = encode(model, [question])
    faiss.normalize_L2(vector)
    # Exact global ordering then metadata filtering is affordable for 5–25 papers,
    # and avoids dropping scoped hits just because unrelated papers rank above them.
    scores, rows = index.search(vector, index.ntotal)
    candidates = [dict(chunks[int(row)], score=float(score))
                  for row, score in zip(rows[0], scores[0])
                  if row >= 0 and chunks[int(row)]['paper_id'] in scope]
    hits = []
    for item in candidates:
        if any(item['paper_id'] == h['paper_id'] and item['page'] == h['page'] and
               max(0, min(item['char_end'], h['char_end']) - max(item['char_start'], h['char_start']))
               / max(1, min(item['char_end']-item['char_start'], h['char_end']-h['char_start'])) > 0.6
               for h in hits):
            continue
        if compare and sum(h['paper_id'] == item['paper_id'] for h in hits) >= 2:
            continue
        hits.append(item)
        if not compare and len(hits) >= k:
            break
        if compare and all(sum(h['paper_id'] == pid for h in hits) >= 2 for pid in scope):
            break
    papers = {p['paper_id']: p for p in collection['manifest']['papers']}
    for i, hit in enumerate(hits, 1):
        paper = papers[hit['paper_id']]
        hit.update(source_id=f'S{i}', title=paper['title'], pdf_url=paper.get('pdf_url', ''), path=paper['path'])
    return hits
