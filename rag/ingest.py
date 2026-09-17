"""Fill the user's original empty ingest.py with page-aware extraction and chunking."""
import hashlib
import re
import unicodedata

import fitz
from config import CHUNK_OVERLAP, CHUNK_TOKENS


def extract_pages(path):
    pages, warnings = [], []
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError('Encrypted PDFs are not supported.')
        if len(doc) > 150:
            raise ValueError('PaperIntel supports PDFs with at most 150 pages.')
        for number, page in enumerate(doc, 1):
            # Blocks preserve local paragraph structure; complex columns still need inspection.
            blocks = page.get_text('blocks', sort=True)
            text = '\n\n'.join(b[4] for b in blocks if len(b) > 6 and b[6] == 0)
            text = unicodedata.normalize('NFKC', text)
            text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
            text = re.sub(r'[ \t]+', ' ', text).strip()
            if len(text) < 100:
                warnings.append(f'PDF page {number} has little extractable text; inspect it manually.')
            pages.append(dict(page=number, text=text))
    if not any(p['text'] for p in pages):
        raise ValueError('No text extracted. This PDF may require OCR, which is not included.')
    return pages, warnings


def chunk_pages(pages, paper_id, tokenizer, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """Use tokenizer offsets to keep original text and exact 1-based PDF page numbers.

    End near a sentence boundary when available. A chunk never crosses a page.
    Token budgets exclude special tokens; caller checks the model's actual limit.
    """
    if max_tokens < 10 or not 0 <= overlap < max_tokens:
        raise ValueError('Chunk size must exceed overlap, with at least 10 tokens.')
    chunks = []
    for page in pages:
        text = page['text']
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False)['offset_mapping']
        start = 0
        while start < len(offsets):
            end = min(start + max_tokens, len(offsets))
            if end < len(offsets):
                for cut in range(end, start + max(max_tokens // 2, overlap + 1), -1):
                    if re.search(r'[.!?][\s\"\u201d]*$', text[offsets[cut - 1][0]:offsets[cut][0]]):
                        end = cut
                        break
            lo, hi = offsets[start][0], offsets[end - 1][1]
            passage = text[lo:hi].strip()
            if passage:
                key = f'{paper_id}:{page["page"]}:{lo}:{hi}'
                chunks.append(dict(chunk_id=hashlib.sha256(key.encode()).hexdigest()[:20],
                                   paper_id=paper_id, page=page['page'], text=passage,
                                   char_start=lo, char_end=hi, token_count=end-start))
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
    return chunks
