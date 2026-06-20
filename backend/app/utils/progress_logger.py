from collections import deque
from datetime import datetime
from threading import Lock
from typing import Deque, Dict, List

_LOGS: Dict[str, Deque[str]] = {}
_LOCK = Lock()
_MAX_LEN = 200


def add_log(video_id: str | None, message: str) -> None:
    """
    为指定 video_id 追加一条调度/进度日志。
    """
    if not video_id:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    with _LOCK:
        q = _LOGS.setdefault(video_id, deque(maxlen=_MAX_LEN))
        q.append(line)


def get_logs(video_id: str) -> List[str]:
    """
    获取指定 video_id 的全部日志（按时间顺序）。
    """
    with _LOCK:
        if video_id not in _LOGS:
            return []
        return list(_LOGS[video_id])

