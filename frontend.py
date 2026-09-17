"""PaperIntel's discovery and evidence-reading interface."""
import html
import json
import os
import re
from pathlib import Path
import streamlit as st
import fitz
from config import ROOT, DATA, QWEN_MODEL, PUBLIC_DEMO
from demo_runtime import session_data, reserve
from paper_fetching import load_model, search_arxiv, build_faiss_index, retrieve_top_papers
from rag.store import Store
from rag.retrieval import retrieve
from rag.generation import generate, build_messages
from rag.guided import GUIDES, guided_question, retrieve_overview

APP_VERSION = '0.4.1'

st.set_page_config(
    page_title="PaperIntel",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",   # sidebar open by default; toggle works natively
)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR CONSTANTS  (hardcoded hex — CSS vars don't resolve in inline styles)
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#f4f7fc"
SURFACE   = "#ffffff"
SURFACE2  = "#edf2fa"
BORDER    = "#d5deed"
INK       = "#172640"
MUTED     = "#465b79"
FAINT     = "#596f8b"
INDIGO    = "#5546d8"
INDIGO_LO = "#ece9ff"
AMBER     = "#805b00"
AMBER_LO  = "#fff4d4"
GREEN     = "#13795b"

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
#  FIX 1: removed  `header { visibility: hidden }`  — that selector also hides
#          the sidebar collapse arrow.  We only hide the Streamlit footer now.
#  FIX 2: all CSS variables are also expressed as hardcoded hex for selectors
#          that Streamlit re-renders inside iframes (metrics, expanders, etc.)
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* TOKENS */
:root {{
    --bg:        {BG};
    --surface:   {SURFACE};
    --surface2:  {SURFACE2};
    --border:    {BORDER};
    --ink:       {INK};
    --muted:     {MUTED};
    --indigo:    {INDIGO};
    --indigo-lo: {INDIGO_LO};
    --amber:     {AMBER};
    --amber-lo:  {AMBER_LO};
    --green:     {GREEN};
}}

/* GLOBAL */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    color: {INK} !important;
}}

.stApp {{
    background:
        radial-gradient(circle at 90% 0%, #e6e5ff, transparent 30rem),
        linear-gradient(180deg, #f9fbff 0%, {BG} 100%) !important;
}}

.block-container {{
    padding-top: 2.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}}

/* Streamlit's fixed application header caused the white strip in the UI. */
header[data-testid="stHeader"] {{
    background: {BG} !important;
    height: 2rem !important;
    border-bottom: none !important;
}}

/* Hide deployment chrome, not the sidebar's reopen/collapse controls. */
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}

[data-testid="stToolbar"], [data-testid="stDecoration"] {{
    background: transparent !important;
    color: {MUTED} !important;
}}

header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] span {{
    color: {MUTED} !important;
}}

/* SIDEBAR  — only colour, no layout override so toggle stays intact */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #eaf0fc 0%, #f1f5fc 100%) !important;
    border-right: 1px solid {BORDER} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {INK} !important;
}}

/* sidebar collapse button — make it visible on dark bg */
button[data-testid="baseButton-headerNoPadding"],
button[kind="header"] {{
    color: {MUTED} !important;
    background: transparent !important;
}}

/* INPUTS */
div[data-testid="stTextInput"] input {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {INK} !important;
    font-family: 'Inter', sans-serif !important;
}}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {{
    color: {FAINT} !important;
    opacity: 1 !important;
}}

div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {{
    background: {SURFACE2} !important;
    border-color: {BORDER} !important;
    color: {INK} !important;
}}

div[data-testid="stNumberInput"] button {{
    background: #e3eafa !important;
    border-color: {BORDER} !important;
    color: {INK} !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {INDIGO} !important;
    box-shadow: 0 0 0 3px {INDIGO_LO} !important;
}}
div[data-testid="stTextInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {{
    color: {MUTED} !important;
    font-size: 0.82rem !important;
}}

/* SELECTBOX */
div[data-testid="stSelectbox"] > div > div {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {INK} !important;
}}

div[data-baseweb="select"] * {{
    color: {INK} !important;
}}

/* SLIDER track */
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {{
    color: {MUTED} !important;
}}

/* BUTTONS — primary */
.stButton > button[kind="primary"],
button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(135deg, #7868ff, #5b8cff) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #8b7eff, #72a0ff) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}}

/* BUTTONS — secondary */
.stButton > button:not([kind="primary"]) {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    color: {INK} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    border-color: {INDIGO} !important;
    color: #818cf8 !important;
}}

/* DOWNLOAD BUTTON */
div[data-testid="stDownloadButton"] > button {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    color: {INK} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    border-color: {INDIGO} !important;
    color: #818cf8 !important;
}}

/* LINK BUTTON */
a[data-testid="stLinkButton"] {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    color: {MUTED} !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}
a[data-testid="stLinkButton"]:hover {{
    border-color: {INDIGO} !important;
    color: #818cf8 !important;
}}

/* METRICS */
div[data-testid="stMetric"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: {INK} !important;
}}
div[data-testid="stMetricLabel"] {{
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
}}

/* TABS */
.stTabs [data-baseweb="tab-list"] {{
    background: #e9edf8 !important;
    border-radius: 14px !important;
    padding: 0.3rem !important;
    border: 1px solid {BORDER} !important;
    gap: 0.25rem !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 10px !important;
    color: {MUTED} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #7868ff, #5b8cff) !important;
    color: #ffffff !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: transparent !important;
}}

/* EXPANDERS */
div[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}}
div[data-testid="stExpander"] summary {{
    color: {MUTED} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    background: {SURFACE} !important;
    padding: 0.75rem 1rem !important;
}}
div[data-testid="stExpander"] summary:hover {{
    color: {INK} !important;
}}

/* ALERTS */
div[data-testid="stAlert"] {{
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}}

/* DIVIDER */
hr {{ border-color: {BORDER} !important; }}

/* PROGRESS */
div[data-testid="stProgressBar"] > div > div {{
    background: {INDIGO} !important;
}}

/* SPINNER TEXT */
div[data-testid="stSpinner"] p {{
    color: {MUTED} !important;
    font-size: 0.88rem !important;
}}

/* CAPTION */
.stCaption, div[data-testid="stCaptionContainer"] {{
    color: {MUTED} !important;
    font-size: 0.8rem !important;
}}

/* MARKDOWN TEXT inside main area */
.stMarkdown p, .stMarkdown li {{
    color: {INK} !important;
}}

/* HIDE only the footer, NOT the header (header has the sidebar toggle) */
footer {{ visibility: hidden !important; }}

/* ── PAPER CARD  (FIX 3: uses only inline styles in Python, no class refs) ── */
/* These classes ARE used — only in st.markdown blocks that are not nested
   inside columns/tabs where the shadow DOM breaks var() resolution.          */
.arxiv-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 0.9rem;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}}
.arxiv-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.45);
    box-shadow: 0 8px 32px rgba(99,102,241,0.1);
}}

/* EYEBROW LABEL */
.eyebrow {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {INDIGO};
    margin-bottom: 0.4rem;
    font-family: 'Space Grotesk', sans-serif;
    display: block;
}}

/* BRAND */
.brand-wrap {{
    margin-bottom: 0.25rem;
}}
.brand-name {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: {INK};
    letter-spacing: -0.02em;
}}
.brand-hex {{
    color: {INDIGO};
    font-size: 1.35rem;
    margin-right: 0.4rem;
}}
.brand-tag {{
    font-size: 0.82rem;
    color: {MUTED};
    margin-bottom: 1.4rem;
    line-height: 1.55;
}}

/* STATUS */
.status-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: {MUTED};
    margin-bottom: 1.25rem;
    font-family: 'Inter', sans-serif;
}}
.dot-live {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {GREEN};
    box-shadow: 0 0 7px {GREEN};
    display: inline-block;
    animation: breathe 2.5s ease-in-out infinite;
    flex-shrink: 0;
}}
.dot-off {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {FAINT};
    display: inline-block;
    flex-shrink: 0;
}}
@keyframes breathe {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.35; }}
}}

/* STEP */
.step-row {{
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-bottom: 0.7rem;
}}
.step-num {{
    flex-shrink: 0;
    width: 1.45rem; height: 1.45rem;
    border-radius: 50%;
    background: {INDIGO_LO};
    color: {INDIGO};
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.05rem;
}}
.step-text {{
    font-size: 0.83rem;
    color: {MUTED};
    line-height: 1.5;
}}
.step-text strong {{ color: {INK}; }}

/* EMPTY STATE */
.empty-state {{
    background: {SURFACE};
    border: 1px dashed rgba(99,102,241,0.2);
    border-radius: 18px;
    padding: 3rem 2rem;
    text-align: center;
}}
.empty-icon  {{ font-size: 1.8rem; margin-bottom: 0.6rem; opacity: 0.45; display: block; }}
.empty-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: {MUTED}; margin-bottom: 0.3rem; display: block; }}
.empty-hint  {{ font-size: 0.84rem; color: {FAINT}; display: block; }}

/* HERO */
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.65rem, 2.5vw, 2.3rem);
    font-weight: 700;
    color: {INK} !important;
    margin: 0;
    padding: 0;
    letter-spacing: -0.03em;
    line-height: 1.2;
}}
.hero-sub {{
    font-size: 0.97rem;
    color: {MUTED} !important;
    margin: 0;
    line-height: 1.65;
}}

/* Stronger hierarchy and readable content throughout the main panel. */
h1, h2, h3, [data-testid="stHeadingWithActionElements"] {{
    color: {INK} !important;
}}

label, p, [data-testid="stCaptionContainer"] {{
    color: {MUTED} !important;
}}

div[data-testid="stForm"] {{
    background: #ffffff !important;
    border: 1px solid {BORDER} !important;
    border-radius: 18px !important;
    padding: 1.15rem 1.25rem !important;
    box-shadow: 0 8px 28px rgba(37,63,100,0.06) !important;
}}

button[kind="primary"] p,
[data-testid="stFormSubmitButton"] button p,
.stTabs [aria-selected="true"] p {{ color: #ffffff !important; }}

div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] > div > div {{ background-color: {SURFACE2} !important; }}

[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{ -webkit-text-fill-color: {INK} !important; }}

[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] h1 {{
    font-size: 1.65rem !important;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_paper_card(paper, show_rank=True):
    paper = dict(paper)
    paper["title"] = html.escape(paper["title"])
    paper["summary"] = html.escape(paper["summary"])
    paper["authors"] = [html.escape(a) for a in paper.get("authors", [])]
    authors = ", ".join(paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors += " et al."

    # Rank pill  — gradient circle, only shown when show_rank=True
    rank_html = ""
    if show_rank and "rank" in paper:
        rank_html = (
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:2rem;height:2rem;border-radius:50%;'
            f'background:linear-gradient(135deg,#6366f1,#a855f7);'
            f'color:#fff;font-family:Space Grotesk,sans-serif;font-weight:700;'
            f'font-size:0.75rem;box-shadow:0 0 10px rgba(99,102,241,0.4);'
            f'margin-right:0.6rem;vertical-align:middle;">#{paper["rank"]}</span>'
        )

    # Relevance badge
    badge_html = ""
    if "similarity" in paper:
        badge_html = (
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem;'
            f'background:{AMBER_LO};color:{AMBER};'
            f'font-size:0.75rem;font-weight:700;font-family:Space Grotesk,sans-serif;'
            f'padding:0.2rem 0.6rem;border-radius:999px;'
            f'border:1px solid rgba(245,158,11,0.22);">'
            f'● {paper["similarity"]:.3f} similarity</span>'
        )

    abstract_preview = paper["summary"][:360]
    if len(paper["summary"]) > 360:
        abstract_preview += "…"

    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:18px;
                    padding:1.2rem 1.4rem;margin-bottom:0.85rem;
                    transition:transform 0.22s ease,border-color 0.22s ease,box-shadow 0.22s ease;">
            <div style="display:flex;align-items:flex-start;gap:0.5rem;margin-bottom:0.65rem;">
                <div style="flex:1;min-width:0;">
                    <p style="font-family:'Space Grotesk',sans-serif;font-size:0.98rem;
                               font-weight:600;color:{INK};margin:0 0 0.3rem 0;
                               line-height:1.4;">{rank_html}{paper["title"]}</p>
                    <p style="font-size:0.82rem;color:{MUTED};margin:0;
                               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {authors}
                    </p>
                </div>
            </div>
            <p style="font-size:0.88rem;color:{MUTED};line-height:1.65;margin:0 0 0.75rem 0;">
                {abstract_preview}
            </p>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner='Loading MiniLM (first run downloads model weights)…')
def get_model():
    return load_model()


@st.cache_data(ttl=900, max_entries=50, show_spinner=False)
def cached_search(query, candidates, category, lo, hi, order):
    return search_arxiv(query, candidates, category, lo, hi, order)


def activate(collection):
    st.session_state.collection = collection
    st.session_state.trace = None
    st.session_state.answer = None


def bundled_paper():
    return dict(paper_id='2005.11401v4',
                title='Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                summary='', authors=['Patrick Lewis et al.'],
                paper_url='https://arxiv.org/abs/2005.11401v4',
                pdf_url='https://arxiv.org/pdf/2005.11401v4',
                local_path=str(ROOT / 'paper' / 'RAG_for_knowledge_intensive_NLP.pdf'))


def pdf_reader(paper, namespace):
    """Full-document page navigation and original-byte download, without embeddings."""
    st.write('**' + paper['title'] + '**')
    if paper.get('authors'):
        st.caption(', '.join(paper['authors']))
    if paper.get('paper_url'):
        st.link_button('Open paper on arXiv ↗', paper['paper_url'])
    pid = paper['paper_id']
    known = st.session_state.pdf_previews.get(pid)
    # Prepared collections already have a local, validated PDF.
    if paper.get('path') and (DATA / paper['path']).is_file():
        known = str(DATA / paper['path'])
    if known and not Path(known).is_file():
        known = None
    if not known:
        st.caption('Load once to browse every page and download the original PDF. No model or token needed.')
        if st.button('Load PDF preview & download', key=namespace + '_load'):
            try:
                with st.spinner('Loading the original PDF…'):
                    if PUBLIC_DEMO:
                        reserve('preview')
                    path, _ = store.pdf_for(paper)
                known = str(path)
                st.session_state.pdf_previews[pid] = known
            except Exception as exc:
                st.error(f'Could not load this PDF: {type(exc).__name__}: {exc}')
    if known:
        try:
            data = Path(known).read_bytes()
            filename = re.sub(r'[^\w.-]+', '_', pid) + '.pdf'
            st.download_button('Download original PDF', data, file_name=filename,
                               mime='application/pdf', key=namespace + '_download')
            with fitz.open(stream=data, filetype='pdf') as doc:
                page_no = int(st.number_input('PDF page', min_value=1, max_value=len(doc), value=1,
                                             step=1, key=namespace + '_page_' + pid))
                st.caption(f'Page {page_no} of {len(doc)} · All pages are available using the page control.')
                zoom = st.select_slider('Page size', options=['Compact', 'Comfortable', 'Large'],
                                        value='Comfortable', key=namespace + '_zoom')
                scale = {'Compact': 1.0, 'Comfortable': 1.5, 'Large': 2.0}[zoom]
                pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(scale, scale))
                st.image(pix.tobytes('png'), caption=f"{paper['title']} · PDF p. {page_no}")
        except Exception as exc:
            st.error(f'PDF preview failed: {type(exc).__name__}: {exc}')


def prepare(papers, name, learning_sample=False):
    with st.status('Preparing your reading collection…', expanded=True) as status:
        try:
            if PUBLIC_DEMO:
                if len(papers) > 5:
                    raise ValueError('The public app supports up to five papers per collection.')
                reserve('prepare')
            collection = store.build(papers, get_model(), name, progress=status.write,
                                     learning_sample=learning_sample)
            activate(collection)
            count = sum(p['status'] == 'ready' for p in collection['manifest']['papers'])
            status.update(label=f'{count}/{len(papers)} papers ready. Open Paper Intelligence.', state='complete')
        except Exception as exc:
            status.update(label='Collection preparation failed', state='error')
            st.error(f'{type(exc).__name__}: {exc}')


def show_evidence(hits):
    for hit in hits:
        with st.expander(f"[{hit['source_id']}] {hit['title'][:85]} · PDF p. {hit['page']} · similarity {hit['score']:.3f}"):
            st.write(hit['text'])
            st.caption(f"Paper ID: {hit['paper_id']} | Chunk: {hit['chunk_id']} | Tokens: {hit['token_count']}")
            if hit.get('pdf_url'):
                st.link_button('Open original at cited page ↗', hit['pdf_url'] + f"#page={hit['page']}")
            if st.button('Show cited PDF page', key='page_' + hit['chunk_id']):
                with fitz.open(DATA / hit['path']) as doc:
                    pix = doc[hit['page'] - 1].get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
                    st.image(pix.tobytes('png'), caption=f"PDF page {hit['page']}")


for key, value in dict(ranked=[], last_query='', collection=None, trace=None, answer=None, pdf_previews={},
                       search_error=None).items():
    if key not in st.session_state:
        st.session_state[key] = value
if PUBLIC_DEMO:
    DATA = session_data(st.session_state)
store = Store(DATA)

with st.sidebar:
    st.title('⬡ PaperIntel')
    st.caption(APP_VERSION)
    st.caption('Find papers. Understand the evidence.')
    if PUBLIC_DEMO:
        st.info('Your collection is private to this browser session and temporary. Download your evidence before leaving.')
    st.write('**Generation:** ' + ('Token configured' if os.getenv('HF_TOKEN') else 'Token not configured'))
    st.caption('Discovery and passage retrieval work without a Qwen token. The first MiniLM load needs internet.')
    if not PUBLIC_DEMO:
        with st.expander('Which version is running?'):
            st.code(str(Path(__file__).resolve()), language=None)
            st.caption('This path identifies the exact copy running on your computer.')
    st.divider()
    saved = store.list_collections()
    if saved:
        selected_saved = st.selectbox('Saved collections', [c['id'] for c in saved],
                                    format_func=lambda x: next(c['name'] + ' · ' + c['created'] for c in saved if c['id'] == x))
        if st.button('Open collection'):
            try:
                activate(store.load(selected_saved))
            except Exception as exc:
                st.error(f'Could not open collection: {exc}')
    st.divider()
    st.write('**Start with your paper**')
    st.caption('Start with one paper, then try a five-paper discovery collection.' if PUBLIC_DEMO else 'A one-paper learning example. Normal discovery collections contain 5–25 papers.')
    if st.button('Prepare bundled RAG paper'):
        sample = bundled_paper()
        prepare([sample], 'Learning example — RAG paper', learning_sample=True)

st.markdown('<h1 class="hero-title">From finding papers to understanding them.</h1>', unsafe_allow_html=True)
st.caption('Discover → Inspect the original PDF → Choose your collection → Understand with cited evidence')
discovery_tab, intelligence_tab = st.tabs(['1 · Paper Discovery', '2 · Paper Intelligence'])

with discovery_tab:
    with st.form('search_form'):
        query = st.text_input('Research topic', placeholder='retrieval augmented generation')
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox('arXiv category (optional)', ['', 'cs.AI', 'cs.CL', 'cs.LG', 'cs.CV', 'cs.IR', 'stat.ML'])
        with col2:
            candidates = st.select_slider('Candidate pool', [25, 50, 100, 200], value=50)
        with col3:
            order = st.selectbox('Candidate ordering', ['relevance', 'submittedDate', 'lastUpdatedDate'])
        use_years = st.checkbox('Filter by submission year')
        col1, col2 = st.columns(2)
        lo = col1.number_input('From year', 1991, 2100, 2020)
        hi = col2.number_input('Through year', 1991, 2100, 2026)
        search_clicked = st.form_submit_button('Search papers', type='primary')
    st.caption('arXiv returns keyword candidates. MiniLM + FAISS ranks only this pool; scores are not probabilities.')
    if search_clicked:
        st.session_state.ranked = []
        st.session_state.pop('basket', None)
        if not query.strip():
            st.warning('Enter a topic first.')
        else:
            try:
                with st.spinner('Fetching candidates and ranking their titles and abstracts…'):
                    if PUBLIC_DEMO:
                        if candidates > 50:
                            raise ValueError('Choose a candidate pool of 25 or 50 for the public app.')
                        reserve('search')
                    papers = cached_search(query.strip(), candidates, category, lo if use_years else None,
                                           hi if use_years else None, order)
                    if papers:
                        model = get_model()
                        index = build_faiss_index(papers, model)
                        st.session_state.ranked = retrieve_top_papers(query, papers, index, model, len(papers))
                        st.session_state.last_query = query
                    else:
                        st.info('No candidates found. Try fewer keywords or broader filters.')
            except Exception as exc:
                st.error(f'Search failed: {type(exc).__name__}: {exc}')
    ranked = st.session_state.ranked
    if ranked:
        st.write(f"**{len(ranked)} ranked candidates** for “{st.session_state.last_query}”")
        by_id = {p['paper_id']: p for p in ranked}
        selected_ids = st.multiselect('Choose 5 papers for your reading collection' if PUBLIC_DEMO else 'Choose 5–25 papers for your reading collection', list(by_id),
            format_func=lambda pid: f"#{by_id[pid]['rank']} · {by_id[pid]['title']}",
            max_selections=5 if PUBLIC_DEMO else 25, key='basket')
        name = st.text_input('Collection name', value=st.session_state.last_query[:80])
        if st.button('Prepare selected papers', type='primary', disabled=not 5 <= len(selected_ids) <= 25):
            prepare([by_id[pid] for pid in selected_ids], name)
        st.caption(f'{len(selected_ids)} selected. PDFs load only when you request a preview or prepare the collection.')
        st.subheader('Inspect before you select')
        inspected_id = st.selectbox('Paper to preview', list(by_id),
                                   format_func=lambda pid: by_id[pid]['title'], key='inspect_discovery')
        with st.expander('Full-paper reader & download', expanded=True):
            pdf_reader(by_id[inspected_id], 'discovery_reader')
        display_count = st.slider('Candidates to display', 1, len(ranked), min(10, len(ranked))) if len(ranked) > 1 else 1
        for paper in ranked[:display_count]:
            render_paper_card(paper)
            with st.expander('Abstract and metadata · ' + paper['paper_id']):
                st.write(paper['summary'])
                st.caption('Submitted: ' + paper['published'][:10] + ' | Categories: ' + ', '.join(paper['categories']))
                st.link_button('Open on arXiv ↗', paper['paper_url'])
    else:
        st.subheader('Try the paper reader first')
        st.caption('Your bundled paper is ready to preview. This step requires no API token or model download.')
        with st.expander('Full-paper reader & download', expanded=True):
            pdf_reader(bundled_paper(), 'sample_reader')

with intelligence_tab:
    collection = st.session_state.collection
    if collection is None:
        st.info('Select 5–25 discovery papers, open a saved collection, or prepare the bundled learning paper.')
    else:
        manifest = collection['manifest']
        ready = {p['paper_id']: p for p in manifest['papers'] if p['status'] == 'ready'}
        st.subheader(manifest['name'])
        a, b, c = st.columns(3)
        a.metric('Papers ready', f"{len(ready)}/{len(manifest['papers'])}")
        b.metric('Extracted pages', sum(p['pages'] for p in ready.values()))
        c.metric('Searchable passages', manifest['chunk_count'])
        with st.expander('Collection preparation details'):
            for paper in manifest['papers']:
                if paper['status'] == 'failed':
                    st.error(paper['title'] + ': ' + paper['error'])
                else:
                    st.write(f"{paper['title']} — {paper['pages']} pages, {paper['chunk_count']} chunks")
                    for warning in paper['warnings']:
                        st.warning(warning)
        st.caption('Citations use 1-based PDF page numbers, which can differ from printed page labels. Tables, equations and figures need manual checking.')
        with st.expander('Read or download a paper in this collection'):
            reader_id = st.selectbox('Collection paper', list(ready),
                                    format_func=lambda pid: ready[pid]['title'], key='reader_' + manifest['id'])
            pdf_reader(ready[reader_id], 'collection_reader_' + manifest['id'])
        st.subheader('Understand the paper')
        mode = st.selectbox('Explanation mode', ['Ask a question', 'Full guided explanation'] + list(GUIDES),
                            key='mode_' + manifest['id'])
        guided = mode != 'Ask a question'
        scope = st.multiselect('Papers to question (empty means all ready papers)', list(ready),
                               format_func=lambda pid: ready[pid]['title'], key='scope_' + manifest['id'], disabled=guided)
        compare_selected = st.checkbox('Compare papers (select 2–5 above)', key='compare_' + manifest['id'], disabled=guided)
        compare = compare_selected and not guided
        if guided:
            guided_id = st.selectbox('Paper to explain', list(ready), format_func=lambda pid: ready[pid]['title'],
                                     key='guided_paper_' + manifest['id'])
            scope = [guided_id]
            question = guided_question(mode)
            st.info(question)
            st.caption('Guided explanations use full-text evidence, not just the abstract. Missing evidence is acknowledged.')
        else:
            question = st.text_area('Your question', placeholder='How do RAG-Sequence and RAG-Token differ?', key='question_' + manifest['id'])
        k = st.slider('Evidence passages', 3, 10, 6, disabled=compare or mode == 'Full guided explanation')
        if compare:
            st.caption('Comparison retrieves up to two passages per selected paper to keep evidence balanced.')
        signature = json.dumps([manifest['id'], question, sorted(scope), compare, k, mode])
        col1, col2 = st.columns(2)
        retrieve_clicked = col1.button('1. Retrieve evidence', type='primary')
        generate_clicked = col2.button('2. Generate cited answer', disabled=not os.getenv('HF_TOKEN'))
        if retrieve_clicked or generate_clicked:
            st.session_state.answer = None
            st.session_state.trace = None
            try:
                if compare and not 2 <= len(scope) <= 5:
                    raise ValueError('Select 2–5 papers explicitly for comparison.')
                with st.spinner('Searching passages…'):
                    if mode == 'Full guided explanation':
                        hits = retrieve_overview(collection, get_model(), guided_id)
                    else:
                        hits = retrieve(question, collection, get_model(), scope or None, k=k, compare=compare)
                    st.session_state.trace = dict(signature=signature, question=question, hits=hits, compare=compare)
                if generate_clicked:
                    if PUBLIC_DEMO:
                        reserve('generation')
                    with st.spinner('Qwen is answering from the retrieved passages…'):
                        st.session_state.answer = generate(question, hits, compare)
            except Exception as exc:
                st.error(f'{type(exc).__name__}: {exc}')
                if generate_clicked:
                    st.caption('If the model/provider is unavailable, update QWEN_MODEL in .env and restart. Retrieved evidence remains available.')
        trace = st.session_state.trace
        if trace and trace['signature'] == signature:
            answer = st.session_state.answer
            if answer:
                st.subheader('Answer')
                if answer['insufficient_evidence']:
                    st.warning('The model reports insufficient evidence for a complete answer.')
                st.markdown(answer['answer'])
                st.caption(answer['validation'])
                export = dict(collection_id=manifest['id'], question=question, model=QWEN_MODEL,
                              answer=answer, sources=trace['hits'])
                st.download_button('Download answer and evidence (.json)', json.dumps(export, ensure_ascii=False, indent=2),
                                   file_name='paperintel-answer.json', mime='application/json')
                sources_text = '\n'.join(f"[{h['source_id']}] {h['title']} — PDF p. {h['page']}" for h in trace['hits'])
                st.download_button('Save explanation (.txt)', answer['answer'] + '\n\nSources\n' + sources_text,
                                   file_name='paperintel-explanation.txt', mime='text/plain')
            st.subheader('Retrieved evidence')
            st.caption('A high score measures similarity to the question. It does not prove a passage answers it.')
            show_evidence(trace['hits'])
            with st.expander('Learning view: exact context sent to Qwen'):
                st.json(build_messages(question, trace['hits'], compare))
            st.download_button('Download retrieval trace (.json)', json.dumps(trace, ensure_ascii=False, indent=2),
                               file_name='paperintel-retrieval.json', mime='application/json')
        elif trace:
            st.info('The question or scope changed. Retrieve again to avoid showing stale evidence.')

st.divider()
st.caption('Discover papers, inspect the original PDF, and trace answers back to page-linked evidence.')
st.caption('Thank you to arXiv for use of its open access interoperability.')
