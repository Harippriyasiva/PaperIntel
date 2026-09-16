import numpy as np
import pytest
from paper.vector_index import NumpyIndex, normalize_L2, write_index, read_index
from demo_runtime import session_data, reserve

def test_portable_ranking_and_reload(tmp_path):
    vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    normalize_L2(vectors)
    index = NumpyIndex(2)
    index.add(vectors)
    path = tmp_path / 'index.faiss'
    write_index(index, path)
    loaded = read_index(path)
    scores, ids = loaded.search(np.array([[1, 0]], dtype=np.float32), 3)
    assert ids.tolist() == [[0, 2, 1]]
    assert np.allclose(scores, [[1, 1/np.sqrt(2), 0]])

def test_sessions_have_separate_storage():
    first, second = {}, {}
    a, b = session_data(first), session_data(second)
    assert a != b
    (a / 'private.txt').write_text('private')
    assert not (b / 'private.txt').exists()
    assert session_data(first) == a
    first['_temporary_workspace'].cleanup()
    second['_temporary_workspace'].cleanup()

def test_global_capacity_and_expiry(monkeypatch):
    import demo_runtime
    from collections import defaultdict, deque
    monkeypatch.setattr(demo_runtime, '_calls', defaultdict(deque))
    for _ in range(20):
        reserve('generation', now=0)
    with pytest.raises(ValueError, match='hourly capacity'):
        reserve('generation', now=3599)
    reserve('generation', now=3600)

def test_public_entry_uses_isolated_storage(monkeypatch):
    import config
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    monkeypatch.setattr(config, 'PUBLIC_DEMO', True)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend.py')).run(timeout=20)
    assert not app.exception
    assert any('private to this browser session' in x.value for x in app.info)
    assert '_temporary_workspace' in app.session_state
