"""Bounded, isolated storage and process-wide quotas for a small public demo."""
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

_lock = threading.Lock()
_calls = defaultdict(deque)
LIMITS = {'generation': 20, 'prepare': 12, 'preview': 60, 'search': 60}

def reserve(action, now=None):
    now = time.monotonic() if now is None else now
    with _lock:
        queue = _calls[action]
        while queue and now - queue[0] >= 3600:
            queue.popleft()
        if len(queue) >= LIMITS[action]:
            raise ValueError('This demo has reached its hourly capacity. Please try again later.')
        queue.append(now)

def session_data(state):
    if '_temporary_workspace' not in state:
        state['_temporary_workspace'] = tempfile.TemporaryDirectory(prefix='paperintel-')
    return Path(state['_temporary_workspace'].name)
