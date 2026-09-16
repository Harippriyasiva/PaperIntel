"""Configuration shared by the UI and application components."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
PUBLIC_DEMO = os.getenv('PAPERINTEL_PUBLIC_DEMO', '0') == '1'
DATA = Path(os.getenv('PAPERINTEL_DATA_DIR', str(ROOT / 'data'))).expanduser().resolve()
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_TOKENS = 200
CHUNK_OVERLAP = 35
SCHEMA_VERSION = 'page-blocks-v1-token200-overlap35'
# Configurable because provider support can change.
QWEN_MODEL = os.getenv('QWEN_MODEL', 'Qwen/Qwen2.5-3B-Instruct:featherless-ai')
HF_BASE_URL = 'https://router.huggingface.co/v1'
