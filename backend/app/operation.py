from threading import Lock
from contextlib import contextmanager
from fastapi import HTTPException
lock = Lock()
@contextmanager
def operation():
    if not lock.acquire(blocking=False):
        raise HTTPException(409, 'Another upload, processing or evaluation is running')
    try:
        yield
    finally:
        lock.release()
