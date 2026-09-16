"""Bounded PDF downloads with content-addressed local storage."""
import hashlib
from pathlib import Path
from urllib.parse import urljoin

import fitz
from paper.discovery import http_session, validate_pdf_url

MAX_PDF_BYTES = 40 * 1024 * 1024


def validate_pdf(data):
    if not data or len(data) > MAX_PDF_BYTES or not data[:1024].lstrip().startswith(b'%PDF-'):
        raise ValueError('Expected a PDF smaller than 40 MB.')
    with fitz.open(stream=data, filetype='pdf') as doc:
        if doc.needs_pass or not len(doc):
            raise ValueError('The PDF is encrypted or empty.')
    return hashlib.sha256(data).hexdigest()


def save_pdf(data, data_dir):
    digest = validate_pdf(data)
    path = Path(data_dir) / 'pdfs' / f'{digest}.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return path, digest


def download_pdf_bytes(url):
    """Validate each redirect rather than following it to an arbitrary host."""
    with http_session() as session:
        for _ in range(5):
            validate_pdf_url(url)
            with session.get(url, timeout=(10, 60), stream=True, allow_redirects=False) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    url = urljoin(url, response.headers.get('Location', ''))
                    continue
                response.raise_for_status()
                pieces, size = [], 0
                for piece in response.iter_content(64 * 1024):
                    size += len(piece)
                    if size > MAX_PDF_BYTES:
                        raise ValueError('PDF exceeds the 40 MB download limit.')
                    pieces.append(piece)
                data = b''.join(pieces)
                validate_pdf(data)
                return data
    raise ValueError('Too many PDF redirects.')
