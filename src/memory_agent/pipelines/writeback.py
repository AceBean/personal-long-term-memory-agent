from datetime import datetime
from ..models.memory_item import MemoryItem


def add_virtual_note_item(index, text: str):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")

    item = MemoryItem(
        id=len(index.items),
        path=f"virtual://{now}",
        modality="virtual",
        timestamp=now,
        preview_text=text[:80],
        meta={"tags": ["virtual"], "content": text},
    )
    vec = index.encode_text([text])
    return item, vec
