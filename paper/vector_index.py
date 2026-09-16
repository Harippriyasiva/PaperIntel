"""Exact inner-product indexes, with a portable NumPy backend if FAISS cannot load."""
import warnings
from pathlib import Path
import numpy as np

try:
    import faiss as _faiss
except (ImportError, OSError):
    _faiss = None

BACKEND = 'faiss' if _faiss is not None else 'numpy'

def normalize_L2(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms != 0)

class NumpyIndex:
    def __init__(self, dimension):
        self.d = int(dimension)
        self.vectors = np.empty((0, self.d), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.d or not np.isfinite(values).all():
            raise ValueError('Invalid vectors for this index.')
        self.vectors = np.concatenate([self.vectors, values])

    def search(self, queries, k):
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.d or not np.isfinite(queries).all():
            raise ValueError('Query dimensions or values are invalid.')
        k = min(max(int(k), 0), self.ntotal)
        scores = queries @ self.vectors.T
        rows = np.argsort(-scores, axis=1, kind='stable')[:, :k]
        return np.take_along_axis(scores, rows, axis=1), rows

def IndexFlatIP(dimension):
    return _faiss.IndexFlatIP(dimension) if _faiss else NumpyIndex(dimension)

def write_index(index, path):
    if isinstance(index, NumpyIndex):
        with open(path, 'wb') as stream:
            np.save(stream, index.vectors, allow_pickle=False)
    else:
        _faiss.write_index(index, str(path))

def read_index(path):
    with open(path, 'rb') as stream:
        numpy_format = stream.read(6) == b'\x93NUMPY'
    if numpy_format:
        with open(path, 'rb') as stream:
            vectors = np.load(stream, allow_pickle=False)
        index = NumpyIndex(vectors.shape[1])
        index.add(vectors)
        return index
    if _faiss is None:
        raise ValueError('This older collection uses a FAISS file unavailable on this system. Prepare it again to create a portable index.')
    return _faiss.read_index(str(path))
