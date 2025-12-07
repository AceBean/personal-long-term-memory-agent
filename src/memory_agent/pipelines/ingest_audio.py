import torch
from memory_agent.models.memory_item import MemoryItem
from memory_agent.utils import file_mtime
from memory_agent.llm.client import qwen_transcribe_audio, qwen_embed_text, qwen_embed_audio_via_summary


def ingest_audio_qwen(index, path: str):
    """
    音频 ingest（Qwen + Whisper 版）：
    1）用本地 Whisper 把音频转成文本
    2）用 Qwen Chat 生成一句话摘要（类似 image caption）
    3）用 text-embedding-v4 将摘要转成 embedding
    4）写入一个 audio 类型 MemoryItem
    """
    # emb_list, summary = qwen_transcribe_audio(path)
    emb_list, summary = qwen_embed_audio_via_summary(path)
    emb = torch.tensor(emb_list, dtype=torch.float32).unsqueeze(0)

    item = MemoryItem(
        id=len(index.items),
        path=path,
        modality="audio",
        timestamp=file_mtime(path),
        preview_text=summary,
        meta={
            "tags": ["audio"],
            "summary": summary,
        },
    )

    return [item], emb