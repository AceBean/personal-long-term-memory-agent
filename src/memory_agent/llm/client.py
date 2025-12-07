import os
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
from typing import List, Tuple
import whisper

# 兼容 OpenAI API 的 DashScope
api_key = "sk-249bef7cbed5492294eb70ba9f3a3de1"
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# ---------------------
# Text Embedding (Qwen)
# ---------------------
def qwen_embed_text(text: str, model: str = "text-embedding-v4", dimensions: int = 1024) -> List[float]:
    """
    使用 text-embedding-v4 做文本向量化。
    - 支持最多 8192 token
    - 维度可配置，默认 1024（官方推荐）:contentReference[oaicite:2]{index=2}
    """
    resp = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions,  # 不写也行，默认 1024
    )
    return resp.data[0].embedding


# -------------------------
# 2. 图像描述：Qwen-VL (qwen3-vl-plus)
# -------------------------


def _prepare_image_for_qwen(img: Image.Image, max_side: int = 768, jpeg_quality: int = 80) -> str:
    """
    将 PIL Image 压缩为较小的 JPEG，并返回 base64 data-url 字符串。
    - 最长边缩放到 max_side
    - JPEG 有损压缩，quality 控制
    """

    # 1. 缩放（保持长宽比）
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 2. 转成 JPEG + 压缩
    buf = BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    data = buf.getvalue()

    # 3. 转成 data-url
    b64 = base64.b64encode(data).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"
    return data_url


def qwen_caption_image(img: Image.Image, model: str = "qwen3-vl-flash") -> str:
    """
    使用通义千问 Qwen-VL 模型对图片生成一句话描述。
    通过 OpenAI 兼容 /chat/completions 接口，模型名示例：qwen3-vl-plus :contentReference[oaicite:3]{index=3}
    """
    data_url = _prepare_image_for_qwen(img, max_side=768, jpeg_quality=80)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请用一句简洁的话详细描述这张图片的内容，用中文回答。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# -------------------------
# 3. 图像 → 文本描述 → 文本向量
# -------------------------
def qwen_embed_image_via_caption(
    img: Image.Image,
    text_model: str = "text-embedding-v4",
    dimensions: int = 1024,
) -> Tuple[List[float], str]:
    """
    “图片 embedding”的实现方式：
    1）先用 Qwen-VL 把图像转成一句话描述 caption
    2）再用 text-embedding-v4 对 caption 做文本向量化
    返回：(embedding, caption)
    """
    caption = qwen_caption_image(img)
    emb = qwen_embed_text(caption, model=text_model, dimensions=dimensions)
    return emb, caption


from typing import List, Tuple

# 如果还没有这个 client，可以参考这样初始化：
# from openai import OpenAI
# openai_client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )


def summarize_video_captions(captions: List[str]) -> str:
    """
    将多帧 caption（通常来自关键帧）总结成一句话的视频摘要。
    captions: ["一张图的描述", "下一张图的描述", ...]
    """
    if not captions:
        return "[视频无有效画面]"

    joined = "\n".join([f"- {c}" for c in captions])

    prompt = (
        "以下是对同一个视频在不同时间点的画面文字描述，请你用**一句简洁的中文句子**"
        "总结这个视频整体的大致内容，不要分点，只要一句话：\n"
        f"{joined}"
    )

    try:
        resp = client.chat.completions.create(
            model="qwen-flash",  # 或你实际开的对话模型名
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"[视频摘要生成失败，使用兜底描述] 该视频包含多个场景，例如：{captions[0][:50]}"

    return summary


def qwen_embed_video_via_summary(captions: List[str]) -> Tuple[List[float], str]:
    """
    视频 → 多帧 caption → 一句话摘要 → 文本 embedding

    参数：
        captions: List[str]，来自关键帧的文字描述列表

    返回：
        emb_list: List[float]，text-embedding-v4 的向量（例如 1024 维）
        summary: str，一句话视频摘要
    """
    if not captions:
        summary = "[视频无有效画面]"
        emb_list = qwen_embed_text(summary)
        return emb_list, summary

    # 1. 用 Qwen Chat 做“一句话总结”
    summary = summarize_video_captions(captions)

    # 2. 用 text-embedding-v4 将摘要编码为向量
    emb_list = qwen_embed_text(summary)

    return emb_list, summary


# -------------------------
# 4. 音频转写（如果你需要）
#    这里只保留接口，具体模型名请按你自己控制台的实际名称改
# -------------------------
def qwen_transcribe_audio(path: str):
    """
    使用本地 Whisper 进行音频转写：
    - 不走网络
    - 不需要 OSS / URL
    - 支持常见音频/视频格式（内部通过 ffmpeg 解码）

    返回：
        segments: [(start_sec, end_sec, text)] 简化为一个整体段
        full_text: 完整识别文本
    """
    try:
        model = whisper.load_model("small")  # 可以改成 base / medium / large
        result = model.transcribe(path, verbose=False)  # 自动语言检测
        full_text = result.get("text", "").strip()
    except Exception as e:
        full_text = f"[Whisper 识别失败] {e}"

    if not full_text:
        full_text = "[音频为空或未识别出有效内容]"

    # 这里我们简单返回一个整体 segment，接口上和之前保持一致
    segments = [(0.0, 0.0, full_text)]
    return segments, full_text


def summarize_text_one_sentence(text: str) -> str:
    """
    用 Qwen Chat 模型把一段文本压缩成一句简洁中文句子
    """
    prompt = f"请将下面这段话总结成一句简洁的中文句子：\n{text}"

    resp = client.chat.completions.create(
        model="qwen-flash",  # 换成你实际开通的 chat 模型
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def qwen_embed_audio_via_summary(path: str) -> Tuple[List[float], str]:
    """
    音频 → Whisper 转写 → Qwen 一句话摘要 → text-embedding-v4

    返回：
        (embedding, summary)
    """
    segments, full_text = qwen_transcribe_audio(path)

    if not full_text or not full_text.strip():
        summary = "[空音频或识别失败]"
        emb = qwen_embed_text(summary)
        return emb, summary

    # 一句话摘要（类似 image caption）
    summary = summarize_text_one_sentence(full_text)

    # 文本向量
    emb = qwen_embed_text(summary)
    return emb, summary


def summarize_video_segment(text: str) -> str:
    prompt = (
            "下面是视频某一段时间内的画面描述，请你先给这段视频取一个简短的小标题，"
            "然后用一句话总结这一段发生了什么。请用中文输出，格式为：标题：xxx；概述：yyy。\n"
            f"{text}"
        )
    try:
        resp = client.chat.completions.create(
            model="qwen-flash",  # 或你实际开的对话模型名
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = "标题：本段视频画面；概述：视频中有一些场景变化。"
    return text