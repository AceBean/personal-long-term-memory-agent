from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class MemoryItem:
    """
    单条记忆条目：
    - 原始文件（图片/文本/视频/音频）
    - 子片段（video_scene/audio_segment）
    - 虚拟 note（chat 写回）
    """
    id: int
    path: str
    modality: str           # "image"/"text"/"video"/"video_scene"/"audio"/"audio_segment"/"virtual"
    timestamp: str          # ISO 时间字符串
    preview_text: str       # 简短摘要/预览
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
