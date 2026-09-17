"""arXiv discovery and MiniLM-based paper ranking."""
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import MODEL_NAME

_NS = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
_API_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def http_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'], respect_retry_after_header=True)
    session.mount('https://', HTTPAdapter(max_retries=retry))
    session.headers['User-Agent'] = 'PaperIntel/0.1 (research paper discovery app)'
    return session


def parse_feed(xml):
    """Keep stable versioned IDs and original metadata for downstream provenance."""
    root = ET.fromstring(xml)
    papers, seen = [], set()
    for entry in root.findall('atom:entry', _NS):
        value = lambda name: ' '.join((entry.findtext('atom:' + name, '', _NS)).split())
        paper_url = value('id').replace('http://', 'https://', 1)
        paper_id = paper_url.split('/abs/')[-1]
        if paper_id.endswith('/errors') or 'api/errors' in paper_url:
            raise ValueError('arXiv rejected the search query. Try simpler keywords.')
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        pdf_url = next((link.attrib.get('href', '') for link in entry.findall('atom:link', _NS)
                        if link.attrib.get('title') == 'pdf' or link.attrib.get('type') == 'application/pdf'), '')
        if not pdf_url:
            pdf_url = paper_url.replace('/abs/', '/pdf/')
        papers.append(dict(paper_id=paper_id, title=value('title'), summary=value('summary'),
                           authors=[' '.join(a.findtext('atom:name', '', _NS).split())
                                    for a in entry.findall('atom:author', _NS)],
                           paper_url=paper_url, pdf_url=pdf_url.replace('http://', 'https://', 1),
                           published=value('published'), updated=value('updated'),
                           categories=[c.attrib.get('term', '') for c in entry.findall('atom:category', _NS)]))
    return papers


def search_arxiv(query, max_results=50, category='', year_from=None, year_to=None, sort_by='relevance'):
    """Candidate retrieval, not a global ranking of all papers on arXiv."""
    global _LAST_REQUEST
    words = re.findall(r'[\w-]+', query, flags=re.UNICODE)
    if not words:
        return []
    if not 1 <= max_results <= 200:
        raise ValueError('Candidate count must be between 1 and 200.')
    if sort_by not in {'relevance', 'submittedDate', 'lastUpdatedDate'}:
        raise ValueError('Unsupported arXiv sort order.')
    # Plain keyword mode avoids interpreting user text as arbitrary query syntax.
    clauses = [f'all:{word}' for word in words]
    if category:
        if not re.fullmatch(r'[A-Za-z-]+(?:\.[A-Za-z]+)?', category):
            raise ValueError('Use an arXiv category such as cs.AI or cs.CL.')
        clauses.append(f'cat:{category}')
    if year_from is not None or year_to is not None:
        lo, hi = int(year_from or 1991), int(year_to or time.gmtime().tm_year)
        if lo < 1991 or hi > 2100 or lo > hi:
            raise ValueError('Check the publication year range.')
        clauses.append(f'submittedDate:[{lo}01010000 TO {hi}12312359]')
    params = dict(search_query=' AND '.join(clauses), start=0, max_results=max_results,
                  sortBy=sort_by, sortOrder='descending')
    # Serialize starts of API calls, including across Streamlit sessions.
    with _API_LOCK:
        time.sleep(max(0.0, 3.0 - (time.monotonic() - _LAST_REQUEST)))
        _LAST_REQUEST = time.monotonic()
        with http_session() as session:
            response = session.get('https://export.arxiv.org/api/query', params=params, timeout=(10, 45))
            response.raise_for_status()
    return parse_feed(response.content)


def load_model():
    from paper.embeddings import MiniLMOnnx
    return MiniLMOnnx(MODEL_NAME)


def encode(model, texts):
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype='float32')
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding output.')
    return np.ascontiguousarray(vectors)


def build_faiss_index(results, model):
    from paper import vector_index as faiss
    if not results:
        raise ValueError('No papers to index.')
    # Retained title+abstract representation; removed the repeated-title heuristic.
    embeddings = encode(model, [p['title'] + '. ' + p['summary'] for p in results])
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_top_papers(query, results, index, model, num_results):
    if not results:
        return []
    from paper import vector_index as faiss
    vector = encode(model, [query])
    faiss.normalize_L2(vector)
    scores, indices = index.search(vector, min(max(int(num_results), 1), len(results)))
    return [dict(results[int(idx)], rank=rank, similarity=round(float(score), 4))
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1) if idx >= 0]


def validate_pdf_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org'}:
        raise ValueError('PDF downloads must use an HTTPS arXiv URL.')
    if parsed.port not in (None, 443) or parsed.username or not parsed.path.startswith('/pdf/'):
        raise ValueError('Invalid arXiv PDF URL.')
    return url
