from PIL import Image
import torch
from memory_agent.models.memory_item import MemoryItem
from memory_agent.utils import file_mtime
from memory_agent.llm.client import qwen_embed_image_via_caption


def ingest_image_qwen(index, path: str):
    """
    图片 ingest 流程（Qwen 版）：
    1. 打开图片
    2. 用 Qwen-VL 生成 caption
    3. 用 text-embedding-v4 对 caption 做 embedding
    4. 存成一个 image 类型的 MemoryItem
    """
    img = Image.open(path).convert("RGB")

    emb_list, caption = qwen_embed_image_via_caption(img)
    emb = torch.tensor(emb_list, dtype=torch.float32).unsqueeze(0)

    item = MemoryItem(
        id=len(index.items),
        path=path,
        modality="image",
        timestamp=file_mtime(path),
        preview_text=caption,
        meta={
            "tags": ["image"],
            "caption": caption,
        },
    )

    return [item], emb
