"""Streamlit can start without credentials or downloading an embedding model."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_ui_starts_without_hf_token(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, 'DATA', tmp_path)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend.py')).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['1 · Paper Discovery', '2 · Paper Intelligence']
    assert any(b.label == 'Prepare example RAG paper' for b in app.button)


def test_sample_retrieval_and_stale_question(monkeypatch, tmp_path):
    import config
    import paper_fetching
    from test_core import Model, make_sample_pdf
    import rag.store
    monkeypatch.setattr(config, 'DATA', tmp_path)
    monkeypatch.setattr(paper_fetching, 'load_model', lambda: Model())
    sample_bytes = make_sample_pdf(tmp_path / 'example.pdf').read_bytes()
    monkeypatch.setattr(rag.store, 'download_pdf_bytes', lambda url: sample_bytes)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend.py')).run(timeout=20)
    next(b for b in app.button if b.label == 'Prepare example RAG paper').click().run(timeout=20)
    assert not app.exception
    assert any(m.label == 'Papers ready' and m.value == '1/1' for m in app.metric)
    app.text_area[0].set_value('How do retrieval token and sequence differ?').run()
    next(b for b in app.button if b.label == '1. Retrieve evidence').click().run(timeout=20)
    assert not app.exception
    assert len(app.session_state['trace']['hits']) == 6
    app.text_area[0].set_value('Another question').run()
    assert any('question or scope changed' in info.value for info in app.info)


def test_five_paper_discovery_selection(monkeypatch, tmp_path):
    import config
    import paper_fetching
    from rag.store import Store
    from test_core import Model, make_sample_pdf
    import rag.store
    monkeypatch.setattr(config, 'DATA', tmp_path)
    monkeypatch.setattr(paper_fetching, 'load_model', lambda: Model())
    sample_bytes = make_sample_pdf(tmp_path / 'example.pdf').read_bytes()
    monkeypatch.setattr(rag.store, 'download_pdf_bytes', lambda url: sample_bytes)
    candidates = [dict(paper_id=str(i), title=f'Retrieval study {i}', summary='retrieval token sequence',
                       authors=['Test Author'], published='2020-01-01', categories=['cs.CL'],
                       paper_url='https://arxiv.org/abs/2005.11401v4',
                       pdf_url='https://arxiv.org/pdf/2005.11401v4') for i in range(5)]
    monkeypatch.setattr(paper_fetching, 'search_arxiv', lambda *args: candidates)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend.py')).run(timeout=20)
    app.text_input[0].set_value('retrieval-test-fixture')
    next(b for b in app.button if b.label == 'Search papers').click().run(timeout=20)
    assert not app.exception
    app.multiselect[0].set_value(['0', '1', '2', '3', '4']).run()
    next(b for b in app.button if b.label == 'Prepare selected papers').click().run(timeout=20)
    assert not app.exception
    assert any(m.label == 'Papers ready' and m.value == '5/5' for m in app.metric)
    assert len(Store(tmp_path).list_collections()) == 1
