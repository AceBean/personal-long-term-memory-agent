import os
import torch
from memory_agent.models.memory_item import MemoryItem
from memory_agent.utils import file_mtime
from memory_agent.llm.client import qwen_embed_text
from memory_agent.pipelines.chunking import split_long_text


def ingest_text_qwen(index, path: str):
    """
    文本 ingest：
    - 自动读取文本
    - 自动 chunking（用于长文本）
    - 每个 chunk 一个子 MemoryItem
    - 父节点为 全文摘要（由 LLM 或简单截断）
    - embedding 使用 Qwen-Embedding-Long（1024维）
    """

    try:
        content = open(path, "r", encoding="utf-8", errors="ignore").read()
    except:
        content = ""

    if not content:
        content = "[空文本内容]"

    # ----------------------
    # 1. chunk 切分
    # ----------------------
    chunks = split_long_text(content, chunk_size=800, overlap=100)

    # 父节点 id
    parent_id = len(index.items)

    # 构建父节点（预览文本）
    preview = content[:120].replace("\n", " ")

    parent_item = MemoryItem(
        id=parent_id,
        path=path,
        modality="text",
        timestamp=file_mtime(path),
        preview_text=preview,
        meta={"tags": ["text"], "chunks": []}
    )

    all_items = [parent_item]
    all_vecs = []

    # ----------------------
    # 2. 子 chunk → embedding
    # ----------------------
    for i, chunk in enumerate(chunks):
        emb = qwen_embed_text(chunk)
        emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        child_item = MemoryItem(
            id=parent_id + 1 + i,
            path=f"{path}#chunk-{i}",
            modality="text_chunk",
            timestamp=file_mtime(path),
            preview_text=chunk[:80],
            meta={"parent_id": parent_id}
        )

        parent_item.meta["chunks"].append(child_item.id)

        all_items.append(child_item)
        all_vecs.append(emb)

    # ----------------------
    # 3. 为父节点生成 embedding（全文平均 embedding）
    # ----------------------
    if all_vecs:
        parent_vec = torch.stack(all_vecs).mean(dim=0)
    else:
        parent_vec = torch.zeros((1, 1024))

    all_vecs.insert(0, parent_vec)

    return all_items, torch.cat(all_vecs, dim=0)
