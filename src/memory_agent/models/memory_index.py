import os
import torch
from typing import List, Dict, Any

from memory_agent.models.memory_item import MemoryItem

# 多模态 ingest 管线
from memory_agent.pipelines.ingest_text import ingest_text_qwen
from memory_agent.pipelines.ingest_image import ingest_image_qwen
from memory_agent.pipelines.ingest_video import ingest_video_qwen
from memory_agent.pipelines.ingest_audio import ingest_audio_qwen

# Qwen 文本 embedding（查询用）
from memory_agent.llm.client import qwen_embed_text

from memory_agent.config import TEXT_EXTS, IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS


class MemoryIndexQwen:
    """
    完整的 Qwen 多模态长期记忆索引系统：
    - 文本使用 Qwen-Embedding-Long（支持 32K tokens）
    - 图片使用 Qwen-VL-Embedding（1024维）
    - 视频场景 caption 用 Qwen-VL + 文本 embedding
    - 音频片段自动转写 + embedding
    """

    # ---------------------------------------------------------
    # 初始化
    # ---------------------------------------------------------
    def __init__(self):
        self.items: List[MemoryItem] = []
        self.embeddings: torch.Tensor | None = None  # shape = (N, 1024)

    # ---------------------------------------------------------
    # 文件分类加入索引
    # ---------------------------------------------------------
    def add_file(self, path: str):
        """根据文件类型选择正确的 ingest pipeline"""

        l = path.lower()

        # try:
        if any(l.endswith(x) for x in TEXT_EXTS):
            items, vecs = ingest_text_qwen(self, path)

        elif any(l.endswith(x) for x in IMAGE_EXTS):
            items, vecs = ingest_image_qwen(self, path)

        elif any(l.endswith(x) for x in VIDEO_EXTS):
            items, vecs = ingest_video_qwen(self, path)

        elif any(l.endswith(x) for x in AUDIO_EXTS):
            items, vecs = ingest_audio_qwen(self, path)

        else:
            print(f"[跳过] 不支持的文件类型：{path}")
            return

        # except Exception as e:
        #     print(f"[错误] ingest 文件失败：{path} | {e}")
        #     return

        # 写入 items
        for it in items:
            self.items.append(it)

        # 写入 embeddings
        if self.embeddings is None:
            self.embeddings = vecs
        else:
            self.embeddings = torch.cat([self.embeddings, vecs], dim=0)

    # ---------------------------------------------------------
    # 批量构建索引
    # ---------------------------------------------------------
    def build_from_folder(self, root: str):
        """递归扫描 root，将所有文件加入索引"""
        if not os.path.exists(root):
            raise FileNotFoundError(f"路径不存在：{root}")

        paths = []
        for base, dirs, files in os.walk(root):
            for f in files:
                paths.append(os.path.join(base, f))

        print(f"[MemoryIndexQwen] 发现 {len(paths)} 个文件，开始 ingest")

        for p in paths:
            self.add_file(p)

        print(f"[MemoryIndexQwen] 索引构建结束，共 {len(self.items)} 条记忆")

    # ---------------------------------------------------------
    # 查询 embedding
    # ---------------------------------------------------------
    def encode_query(self, text: str) -> torch.Tensor:
        """将用户查询转换为 1024维 embedding"""
        vec = qwen_embed_text(text)
        vec = torch.tensor(vec, dtype=torch.float32)
        return vec

    # ---------------------------------------------------------
    # 简单向量检索（扁平）
    # ---------------------------------------------------------
    @torch.no_grad()
    def search(self, query: str, top_k: int = 8):
        """返回最相似的 top_k 结果"""
        q = self.encode_query(query)

        sims = torch.softmax(100 * self.embeddings @ q, dim=0) # 内积
        vals, idxs = torch.topk(sims, k=min(top_k, len(self.items)))

        results = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            results.append({
                "item": self.items[idx],
                "score": score
            })

        return results

    # ---------------------------------------------------------
    # 层级聚合检索（父节点 + 子节点）
    # ---------------------------------------------------------
    @torch.no_grad()
    def search_grouped(self, query: str, top_k: int = 8, max_children: int = 3):
        """
        返回结构如下：
        [
            {
                "parent": MemoryItem,
                "parent_score": float,
                "children": [(MemoryItem, score), ...]
            }
        ]
        """
        flat = self.search(query, top_k=top_k)

        groups = {}

        for hit in flat:
            item = hit["item"]
            score = hit["score"]

            # 子节点？
            pid = item.meta.get("parent_id", item.id)

            if pid not in groups:
                groups[pid] = {
                    "parent": None,
                    "parent_score": score,
                    "children": []
                }

            if item.meta.get("parent_id") == pid:
                groups[pid]["children"].append((item, score))
            else:
                groups[pid]["parent"] = item
                groups[pid]["parent_score"] = score

        # 未命中的父节点处理
        for pid, g in groups.items():
            if g["parent"] is None:
                g["parent"] = g["children"][0][0]

            g["children"] = sorted(g["children"], key=lambda x: x[1], reverse=True)[:max_children]

        # 按父节点排序
        sorted_groups = sorted(groups.values(), key=lambda x: x["parent_score"], reverse=True)
        return sorted_groups

    # ---------------------------------------------------------
    # 构建给 LLM 的多模态上下文
    # ---------------------------------------------------------
    def build_grouped_llm_context(self, query: str, groups: List[Dict[str, Any]]) -> str:
        """
        构造结构化文本，用于 RAG（提供给 Qwen Chat）
        """
        out = []
        out.append(f"用户查询：{query}\n")
        out.append("下面是从多模态记忆库中检索到的最相关的信息：\n")

        for rank, g in enumerate(groups, 1):
            parent = g["parent"]
            out.append(f"\n[{rank}] 模态：{parent.modality}")
            out.append(f"路径：{parent.path}")
            out.append(f"摘要：{parent.preview_text}\n")

            # 展示视频/音频子片段
            for child, sc in g["children"]:
                out.append(f"  - 子片段({child.modality})：{child.preview_text}")

            out.append("\n")

        return "\n".join(out)

    # ---------------------------------------------------------
    # 保存 / 加载
    # ---------------------------------------------------------
    def save(self, path: str):
        data = {
            "items": [it.to_dict() for it in self.items],
            "embeddings": self.embeddings.cpu().numpy()
        }
        torch.save(data, path)
        print(f"[OK] 索引已保存：{path}")

    @classmethod
    def load(cls, path: str):
        d = torch.load(path, map_location="cpu", weights_only=False)
        idx = cls()

        idx.items = [MemoryItem(**raw) for raw in d["items"]]
        idx.embeddings = torch.tensor(d["embeddings"], dtype=torch.float32)

        print(f"[OK] 索引已加载，共 {len(idx.items)} 条记忆")
        return idx