"""Public deployment entry point. Local persistent mode uses frontend.py."""
import os
import runpy
from pathlib import Path
os.environ.setdefault('PAPERINTEL_PUBLIC_DEMO', '1')
runpy.run_path(str(Path(__file__).with_name('frontend.py')), run_name='__main__')
