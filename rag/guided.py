"""Guided research questions backed by full-text evidence."""
from rag.retrieval import retrieve

GUIDES = {
    'Problem Statement': 'What problem does this paper address, and why is it important?',
    'Methodology': 'Explain the proposed methodology, model components and experimental setup.',
    'Key Contributions': 'What are the main contributions and reported findings of this paper?',
    'Advantages': 'What advantages are supported by the experiments or comparisons in this paper?',
    'Limitations': 'What limitations and failure cases do the authors explicitly report?',
    'Future Scope': 'What future work or open questions do the authors explicitly identify?',
}


def guided_question(mode):
    if mode == 'Full guided explanation':
        return ('Explain this paper for a college student under these six headings: '
                + ', '.join(GUIDES)
                + '. Cite evidence for claims. For any heading without supporting evidence, '
                'say it is not established in the retrieved passages. Do not invent future work.')
    return GUIDES[mode] + ' Explain clearly and cite the supplied evidence. Do not infer missing details.'


def retrieve_overview(collection, model, paper_id):
    # Each heading gets a retrieval query; a generic summary query could miss limitations.
    hits, seen = [], set()
    for heading, question in GUIDES.items():
        for hit in retrieve(question, collection, model, [paper_id], k=2):
            if hit['chunk_id'] in seen:
                continue
            seen.add(hit['chunk_id'])
            hits.append(dict(hit, retrieval_topic=heading))
    for number, hit in enumerate(hits, 1):
        hit['source_id'] = f'S{number}'
    return hits
