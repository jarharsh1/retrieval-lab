"""
Metrics utilities — latency tracking and retrieval quality scoring.
"""

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """
    Context manager to measure elapsed time in milliseconds.

    Usage:
        with timer() as t:
            do_something()
        print(f"Took {t.elapsed_ms:.2f}ms")
    """
    t = type("Timer", (), {"elapsed_ms": 0.0})()
    start = time.perf_counter()
    yield t
    t.elapsed_ms = (time.perf_counter() - start) * 1000
