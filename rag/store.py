"""SQLite catalog plus immutable, content-identified FAISS collection directories."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np

from config import MODEL_NAME, SCHEMA_VERSION, CHUNK_TOKENS
from paper.discovery import encode
from paper.documents import download_pdf_bytes, save_pdf, validate_pdf
from rag.ingest import extract_pages, chunk_pages


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS documents (paper_id TEXT PRIMARY KEY, sha TEXT, path TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, name TEXT, manifest TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP)')

    def connect(self):
        return sqlite3.connect(self.root / 'catalog.sqlite3', timeout=30)

    def pdf_for(self, paper):
        if paper.get('local_path'):
            path, sha = save_pdf(Path(paper['local_path']).read_bytes(), self.root)
        else:
            with self.connect() as db:
                row = db.execute('SELECT sha,path FROM documents WHERE paper_id=?', (paper['paper_id'],)).fetchone()
            if row and (self.root / row[1]).is_file():
                path = self.root / row[1]
                if validate_pdf(path.read_bytes()) == row[0]:
                    return path, row[0]
            path, sha = save_pdf(download_pdf_bytes(paper['pdf_url']), self.root)
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO documents VALUES (?,?,?)',
                       (paper['paper_id'], sha, str(path.relative_to(self.root))))
        return path, sha

    def list_collections(self):
        with self.connect() as db:
            return [dict(id=r[0], name=r[1], created=r[2]) for r in
                    db.execute('SELECT id,name,created FROM collections ORDER BY created DESC')]

    def load(self, collection_id):
        from paper import vector_index as faiss
        with self.connect() as db:
            row = db.execute('SELECT manifest FROM collections WHERE id=?', (collection_id,)).fetchone()
        if not row:
            raise ValueError('Collection not found.')
        manifest = json.loads(row[0])
        if manifest['model'] != MODEL_NAME or manifest['schema'] != SCHEMA_VERSION:
            raise ValueError('Embedding/extraction settings changed. Prepare this collection again.')
        directory = self.root / 'collections' / collection_id
        chunks = json.loads((directory / 'chunks.json').read_text(encoding='utf-8'))
        index = faiss.read_index(str(directory / 'index.faiss'))
        if index.ntotal != len(chunks) or index.d != manifest['dimension']:
            raise ValueError('Collection index and metadata do not match. Rebuild the collection.')
        return dict(manifest=manifest, chunks=chunks, index=index)

    def build(self, papers, model, name, progress=lambda message: None, learning_sample=False):
        """Only publish a collection after both metadata and index have been written.

        Failed papers remain explicit in the manifest; successful papers stay usable.
        A one-paper exception exists only for the example learning paper.
        """
        from paper import vector_index as faiss
        if len({p['paper_id'] for p in papers}) != len(papers):
            raise ValueError('Select unique papers.')
        if not (5 <= len(papers) <= 25 or learning_sample and len(papers) == 1):
            raise ValueError('Select 5–25 papers, or use the one-paper learning example.')
        special = model.tokenizer.num_special_tokens_to_add(pair=False)
        if CHUNK_TOKENS + special > model.max_seq_length:
            raise ValueError('Chunk size exceeds the embedding model input limit.')
        outcomes, all_chunks = [], []
        for i, paper in enumerate(papers, 1):
            progress(f'Preparing paper {i}/{len(papers)}: {paper["title"][:70]}')
            clean = {k: v for k, v in paper.items() if k not in {'local_path', 'rank', 'similarity'}}
            try:
                path, sha = self.pdf_for(paper)
                pages, warnings = extract_pages(path)
                chunks = chunk_pages(pages, paper['paper_id'], model.tokenizer)
                if not chunks:
                    raise ValueError('No usable text chunks.')
                clean.update(status='ready', sha=sha, path=str(path.relative_to(self.root)),
                             pages=len(pages), chunk_count=len(chunks), warnings=warnings)
                all_chunks.extend(chunks)
            except Exception as exc:
                clean.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            outcomes.append(clean)
        if not all_chunks:
            errors = '; '.join(p['title'][:35] + ': ' + p.get('error', '') for p in outcomes)
            raise ValueError('No paper could be prepared. ' + errors)
        signature = json.dumps(dict(papers=[(p['paper_id'], p.get('sha'), p['status']) for p in outcomes],
                                    model=MODEL_NAME, schema=SCHEMA_VERSION, backend=faiss.BACKEND), sort_keys=True)
        collection_id = hashlib.sha256(signature.encode()).hexdigest()[:24]
        with self.connect() as db:
            exists = db.execute('SELECT 1 FROM collections WHERE id=?', (collection_id,)).fetchone()
        if exists:
            progress('Reusing the saved collection index.')
            return self.load(collection_id)
        progress(f'Embedding {len(all_chunks)} passages and saving the collection…')
        vectors = encode(model, [c['text'] for c in all_chunks])
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        manifest = dict(id=collection_id, name=name.strip() or 'Paper collection', model=MODEL_NAME,
                        schema=SCHEMA_VERSION, dimension=index.d, papers=outcomes,
                        chunk_count=len(all_chunks), learning_sample=learning_sample)
        parent = self.root / 'collections'
        parent.mkdir(exist_ok=True)
        temporary = parent / ('.building-' + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            (temporary / 'chunks.json').write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding='utf-8')
            (temporary / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            faiss.write_index(index, str(temporary / 'index.faiss'))
            target = parent / collection_id
            try:
                temporary.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
                # Another local session may have completed this identical immutable build.
                shutil.rmtree(temporary)
            with self.connect() as db:
                db.execute('INSERT OR IGNORE INTO collections (id,name,manifest) VALUES (?,?,?)',
                           (collection_id, manifest['name'], json.dumps(manifest)))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.load(collection_id)
