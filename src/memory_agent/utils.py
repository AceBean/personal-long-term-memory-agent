import os
from datetime import datetime

def file_mtime(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")