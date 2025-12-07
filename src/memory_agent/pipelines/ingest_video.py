# import os
# import cv2
# import torch
# from PIL import Image
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector

# from memory_agent.models.memory_item import MemoryItem
# from memory_agent.config import CACHE_DIR
# from memory_agent.utils import file_mtime
# from memory_agent.llm.client import qwen_caption_image, qwen_embed_text


# def detect_scenes(video_path):
#     vm = VideoManager([video_path])
#     sm = SceneManager()
#     sm.add_detector(ContentDetector(threshold=27))
#     vm.start()
#     sm.detect_scenes(frame_source=vm)
#     scenes = sm.get_scene_list()
#     return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


# def extract_frame(path, sec):
#     cap = cv2.VideoCapture(path)
#     cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     ret, frame = cap.read()
#     if not ret:
#         return None
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(img)


# def ingest_video_qwen(index, path: str):
#     scenes = detect_scenes(path)
#     parent_id = len(index.items)

#     parent = MemoryItem(
#         id=parent_id,
#         path=path,
#         modality="video",
#         timestamp=file_mtime(path),
#         preview_text="[视频摘要生成中]",
#         meta={"scenes": []}
#     )

#     all_items = [parent]
#     scene_embeddings = []

#     captions = []

#     for i, (start, end) in enumerate(scenes):
#         mid = (start + end) / 2
#         img = extract_frame(path, mid)
#         if img is None:
#             continue

#         # 保存关键帧
#         os.makedirs(CACHE_DIR, exist_ok=True)
#         kpath = os.path.join(CACHE_DIR, f"{parent_id}_scene_{i}.jpg")
#         img.save(kpath)

#         # Caption
#         caption = qwen_caption_image(img)
#         captions.append(caption)

#         # Embedding
#         emb = qwen_embed_text(caption)
#         emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
#         scene_embeddings.append(emb)

#         child = MemoryItem(
#             id=parent_id + 1 + i,
#             path=f"{path}#scene-{i}",
#             modality="video_scene",
#             timestamp=file_mtime(path),
#             preview_text=caption[:80],
#             meta={
#                 "parent_id": parent_id,
#                 "start_sec": start,
#                 "end_sec": end,
#                 "keyframe_path": kpath
#             }
#         )

#         parent.meta["scenes"].append(child.id)
#         all_items.append(child)

#     # 父节点 embedding = 所有场景 embedding 平均
#     if scene_embeddings:
#         parent_emb = torch.stack(scene_embeddings).mean(dim=0)
#         parent.preview_text = " | ".join(captions[:3])
#         parent.meta["summary"] = "\n".join(captions)
#     else:
#         parent_emb = torch.zeros((1, 1024))

#     all_vecs = [parent_emb] + scene_embeddings
#     return all_items, torch.cat(all_vecs, dim=0)


import os
import cv2
import torch
from PIL import Image

from memory_agent.models.memory_item import MemoryItem
from memory_agent.utils import file_mtime
from memory_agent.llm.client import (
    qwen_caption_image,
    qwen_embed_video_via_summary,
    summarize_text_one_sentence,
    summarize_video_segment
)

# 可选依赖：OCR & YOLO。没有的话会自动降级。
try:
    import pytesseract
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

try:
    from ultralytics import YOLO
    _yolo_model = None
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False
    _yolo_model = None


# ------------------------------------------------------------
# 1. 自适应采样策略
# ------------------------------------------------------------
def get_sampling_strategy(video_seconds: float):
    """
    根据视频时长决定关键帧间隔 & 最大数量
    """
    if video_seconds <= 120:          # <= 2 min
        return 2.0, 12
    elif video_seconds <= 600:        # <= 10 min
        return 4.0, 18
    elif video_seconds <= 1800:       # <= 30 min
        return 8.0, 24
    else:
        return 12.0, 30


# ------------------------------------------------------------
# 2. 关键帧提取（含数量限制）
# ------------------------------------------------------------
def extract_keyframes_adaptive(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_seconds = frame_count / fps if frame_count > 0 else 0.0

    interval_sec, max_frames = get_sampling_strategy(video_seconds)
    frame_interval = max(1, int(interval_sec * fps))

    save_dir = os.path.join(os.path.dirname(path), ".keyframes")
    os.makedirs(save_dir, exist_ok=True)

    keyframes = []
    i = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break

        if i % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            out_path = os.path.join(save_dir, f"{os.path.basename(path)}_kf_{i}.jpg")
            img.save(out_path)

            ts = i / fps
            keyframes.append((ts, out_path, img))
            saved += 1

        i += 1

    cap.release()
    return keyframes


# ------------------------------------------------------------
# 3. 可选 OCR
# ------------------------------------------------------------
def run_ocr_on_image(img: Image.Image) -> str:
    if not _HAS_OCR:
        return ""
    try:
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text.strip()
    except Exception:
        return ""


# ------------------------------------------------------------
# 4. 可选 YOLO 物体检测
# ------------------------------------------------------------
def get_yolo_model():
    global _yolo_model
    if not _HAS_YOLO:
        return None
    if _yolo_model is None:
        # 使用轻量模型，第一次会自动下载
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def run_yolo_on_image(img: Image.Image, conf_thres: float = 0.35):
    model = get_yolo_model()
    if model is None:
        return []

    # try:
    results = model(img, verbose=False)
    objects = set()
    for r in results:
        boxes = r.boxes
        names = r.names
        for box in boxes:
            c = int(box.cls[0].item())
            prob = float(box.conf[0].item())
            if prob >= conf_thres:
                objects.add(names[c])
    return sorted(list(objects))
    # except Exception:
    #     return []


# ------------------------------------------------------------
# 5. Timeline summary（结构化时间线）
# ------------------------------------------------------------
def build_timeline_summary(keyframes, captions):
    """
    构造视频时间线摘要文本（00:00 → xxx）
    """
    lines = []
    for (ts, _, _), cap in zip(keyframes, captions):
        mm = int(ts // 60)
        ss = int(ts % 60)
        t = f"{mm:02d}:{ss:02d}"
        lines.append(f"{t} → {cap}")

    timeline = "\n".join(lines)
    summary_of_timeline = summarize_text_one_sentence(
        "下面是视频关键帧时间线：\n" + timeline
    )
    return timeline, summary_of_timeline


# ------------------------------------------------------------
# 6. 章节划分（简单按时间分段 + 语义总结）
# ------------------------------------------------------------
def build_video_chapters(keyframes, captions, num_chapters: int = 4):
    """
    根据关键帧按时间粗略划分章节，并对每个章节做一句话总结。
    返回格式：
    [
        {
            "start": float 秒,
            "end": float 秒,
            "title": str,
            "summary": str,
        },
        ...
    ]
    """
    if not keyframes:
        return []

    # 章节数量不要超过关键帧数
    num_chapters = max(1, min(num_chapters, len(keyframes)))

    # 按时间排序
    keyframes_sorted = sorted(list(zip(keyframes, captions)), key=lambda x: x[0][0])
    timestamps = [kf[0][0] for kf in keyframes_sorted]
    t_start = timestamps[0]
    t_end = timestamps[-1] if timestamps[-1] > t_start else t_start + 1e-3

    # 为每个章节划定时间区间
    chapters = []
    for i in range(num_chapters):
        s = t_start + (t_end - t_start) * i / num_chapters
        e = t_start + (t_end - t_start) * (i + 1) / num_chapters
        chapters.append({"start": s, "end": e, "captions": []})

    # 把关键帧分配到章节
    for (kf, cap) in keyframes_sorted:
        ts = kf[0]
        # 找对应章节
        for ch in chapters:
            if ch["start"] <= ts <= ch["end"]:
                ch["captions"].append(cap)
                break

    # 对每个章节做标题 + summary
    final_chapters = []
    for ch in chapters:
        caps = ch["captions"]
        if not caps:
            continue

        joined = "\n".join(f"- {c}" for c in caps)
        text = summarize_video_segment(joined)

        final_chapters.append({
            "start": ch["start"],
            "end": ch["end"],
            "raw": text,
        })

    return final_chapters


# ------------------------------------------------------------
# 7. ingest 主函数（整合所有增强）
# ------------------------------------------------------------
def ingest_video_qwen(index, path: str):
    """
    Qwen 视频 ingest（增强版）：
    video → keyframes → caption + OCR + YOLO → timeline → chapters → summary → embedding
    """

    # 1) 自适应关键帧提取
    keyframes = extract_keyframes_adaptive(path)

    if not keyframes:
        from memory_agent.llm.client import qwen_embed_text
        summary = "[视频无法提取关键帧]"
        emb = torch.tensor(qwen_embed_text(summary), dtype=torch.float32).unsqueeze(0)

        item = MemoryItem(
            id=len(index.items),
            path=path,
            modality="video",
            timestamp=file_mtime(path),
            preview_text=summary,
            meta={
                "summary": summary,
                "captions": [],
                "keyframes": [],
                "timeline": "",
                "timeline_summary": "",
                "chapters": [],
            },
        )
        return [item], emb

    # 2) 对每帧做 Qwen-VL caption + OCR + YOLO
    captions = []
    kf_meta = []

    for ts, frame_path, img in keyframes:
        caption = qwen_caption_image(img)
        ocr_text = run_ocr_on_image(img)
        objects = run_yolo_on_image(img)

        captions.append(caption)

        kf_meta.append({
            "timestamp": ts,
            "frame_path": frame_path,
            "caption": caption,
            "ocr_text": ocr_text,
            "objects": objects,
        })

    # 3) 时间线 + 时间线摘要
    timeline_text, timeline_summary = build_timeline_summary(keyframes, captions)

    # 4) 视频整体“一句话摘要” + embedding
    emb_list, video_summary = qwen_embed_video_via_summary(captions)
    emb = torch.tensor(emb_list, dtype=torch.float32).unsqueeze(0)

    # 5) 章节划分（基于 caption + 时间）
    chapters = build_video_chapters(keyframes, captions, num_chapters=4)

    # 6) 构建 MemoryItem
    item = MemoryItem(
        id=len(index.items),
        path=path,
        modality="video",
        timestamp=file_mtime(path),
        preview_text=video_summary,
        meta={
            "summary": video_summary,
            "captions": captions,
            "keyframes": kf_meta,
            "timeline": timeline_text,
            "timeline_summary": timeline_summary,
            "chapters": chapters,
        },
    )

    return [item], emb
