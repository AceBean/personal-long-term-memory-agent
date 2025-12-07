import os
import math
import streamlit as st

from memory_agent.models.memory_index import MemoryIndexQwen
from memory_agent.config import UPLOAD_DIR


# ========== 基本设置 ==========

st.set_page_config(
    page_title="Personal Long-Term Memory Agent (Qwen)",
    page_icon="🧠",
    layout="wide",
)

if "memory_index" not in st.session_state:
    st.session_state.memory_index: MemoryIndexQwen | None = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "你是一个个人长期记忆助手，可以访问用户的多模态记忆（文本、图片、视频、音频、虚拟笔记）。"
        "系统已经为你检索好了和问题相关的记忆快照，请基于这些信息进行回答。"
        "如果记忆不足以支持确定结论，请明确说明不确定。"
    )


# ========== 工具函数 ==========

def ensure_index_loaded() -> bool:
    if st.session_state.memory_index is None:
        st.warning("⚠️ 请先在左侧加载索引文件（.pt）")
        return False
    return True


def call_qwen_chat(messages, model_name: str):
    from openai import OpenAI
    client = OpenAI(
        api_key="sk-249bef7cbed5492294eb70ba9f3a3de1",
        # api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def render_video_meta(parent):
    """
    在检索结果中展示视频特有的信息：
    - 视频播放器
    - 一句话摘要
    - 时间线（timeline）
    - 章节列表
    - 关键帧画廊（含 caption / OCR / objects）
    """
    meta = parent.meta or {}

    # 1. 视频播放器
    if os.path.exists(parent.path):
        st.video(parent.path)

    # 2. 一句话总结
    if "summary" in meta:
        st.markdown(f"**视频摘要：** {meta['summary']}")

    # 3. 时间线 & 时间线摘要
    if "timeline_summary" in meta:
        with st.expander("🕒 时间线总结（Timeline Summary）"):
            st.write(meta["timeline_summary"])

    if "timeline" in meta and meta["timeline"]:
        with st.expander("🧬 关键帧时间线（Timeline）"):
            st.text(meta["timeline"])

    # 4. 章节列表
    chapters = meta.get("chapters", [])
    if chapters:
        with st.expander("📚 视频章节（Chapters）"):
            for i, ch in enumerate(chapters, 1):
                start = ch.get("start", 0.0)
                end = ch.get("end", 0.0)
                mm_s = f"{int(start // 60):02d}:{int(start % 60):02d}"
                mm_e = f"{int(end // 60):02d}:{int(end % 60):02d}"
                st.markdown(f"**第{i}章**  ({mm_s} ~ {mm_e})")
                st.markdown(ch.get("raw", "（无章节内容）"))
                st.markdown("---")

    # 5. 关键帧画廊
    keyframes = meta.get("keyframes", [])
    if keyframes:
        with st.expander("🖼 关键帧画廊（Keyframes Gallery）"):
            cols_per_row = 3
            rows = math.ceil(len(keyframes) / cols_per_row)
            for r in range(rows):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx = r * cols_per_row + c
                    if idx >= len(keyframes):
                        break
                    kf = keyframes[idx]
                    with cols[c]:
                        ts = kf.get("timestamp", 0.0)
                        mm = int(ts // 60)
                        ss = int(ts % 60)
                        t = f"{mm:02d}:{ss:02d}"

                        fp = kf.get("frame_path", "")
                        if os.path.exists(fp):
                            st.image(fp, caption=f"t={t}")

                        st.caption(kf.get("caption", ""))
                        if kf.get("ocr_text"):
                            with st.expander("OCR 文本"):
                                st.text(kf["ocr_text"])
                        if kf.get("objects"):
                            st.caption("检测到物体: " + ", ".join(kf["objects"]))


def render_grouped_results(groups: list[dict]):
    if not groups:
        st.info("没有检索结果。")
        return

    for rank, g in enumerate(groups, start=1):
        parent = g["parent"]
        children = g["children"]
        score = g["parent_score"]

        with st.container(border=True):
            st.markdown(f"### #{rank} | 模态：{parent.modality} | 相似度：{score:.3f}")
            st.text(f"路径: {parent.path}")
            st.text(f"时间: {parent.timestamp}")
            st.markdown(f"**摘要：** {parent.preview_text}")

            if "summary" in parent.meta:
                with st.expander("展开完整摘要"):
                    st.write(parent.meta["summary"])

            # 媒体展示（原版本）
            if parent.modality == "image" and os.path.exists(parent.path):
                st.image(parent.path)
            elif parent.modality == "video":
                render_video_meta(parent)
            elif parent.modality == "audio" and os.path.exists(parent.path):
                st.audio(parent.path)

            # 子节点
            if children:
                st.markdown("**相关子片段：**")
                for child, sc in children:
                    st.markdown(f"- {child.modality} | 相似度 {sc:.3f}")
                    st.markdown(f"  内容：{child.preview_text}")
                    if "keyframe_path" in child.meta:
                        kf = child.meta["keyframe_path"]
                        if os.path.exists(kf):
                            st.image(kf, caption="关键帧", use_container_width=True)

        st.markdown("---")


# ========== Sidebar ==========

st.sidebar.header("⚙ 控制台")

index_path = st.sidebar.text_input("索引文件路径 (.pt)", value="memory_index.pt")

if st.sidebar.button("加载索引"):
    if not os.path.exists(index_path):
        st.sidebar.error(f"索引文件不存在：{index_path}")
    else:
        with st.spinner("正在加载索引..."):
            idx = MemoryIndexQwen.load(index_path)
        st.session_state.memory_index = idx
        st.sidebar.success("索引加载完成！")

if st.session_state.memory_index is not None:
    st.sidebar.markdown(f"**记忆条目数：** {len(st.session_state.memory_index.items)}")

st.sidebar.markdown("---")

llm_model = st.sidebar.text_input(
    "Qwen 对话模型名",
    value="qwen-flash",  # 这里建议你换成自己实际可用的模型名
)

top_k = st.sidebar.slider("检索 top-k 父节点", 3, 20, 8)
max_children = st.sidebar.slider("每个父节点展示子节点数量", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("📤 上传新文件（增量更新索引）")

uploaded = st.sidebar.file_uploader(
    "选择文件（文本/图片/音频/视频）",
    type=["txt", "md", "jpg", "jpeg", "png", "mp4", "mov", "avi", "mp3", "wav", "m4a"],
)

if uploaded is not None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.read())

    st.sidebar.success(f"已保存到 {save_path}")

    if st.sidebar.button("加入记忆库"):
        if not ensure_index_loaded():
            st.stop()
        with st.spinner("正在处理并更新索引..."):
            st.session_state.memory_index.add_file(save_path)
            st.session_state.memory_index.save(index_path)
        st.sidebar.success("已加入索引！")


# ========== Tabs ==========

tab_search, tab_chat = st.tabs(["🔍 检索模式", "💬 聊天模式"])


# ---------- Tab 1: Search ----------
with tab_search:
    st.header("🔍 检索你的多模态记忆库（Qwen）")

    query = st.text_input("输入查询内容：", key="search_query")

    if st.button("执行检索"):
        if not ensure_index_loaded():
            st.stop()
        with st.spinner("检索中..."):
            groups = st.session_state.memory_index.search_grouped(
                query, top_k=top_k, max_children=max_children
            )
        render_grouped_results(groups)


# ---------- Tab 2: Chat ----------
with tab_chat:
    st.header("💬 带记忆检索的聊天")

    # 展示历史消息
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("说点什么...")

    if user_input:
        if not ensure_index_loaded():
            st.stop()

        # 1. 先把用户输入加入聊天记录并展示
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. 检索相关记忆
        idx = st.session_state.memory_index
        with st.spinner("检索相关记忆..."):
            grouped = idx.search_grouped(user_input, top_k=top_k, max_children=max_children)
            mem_ctx = idx.build_grouped_llm_context(user_input, grouped)

        # 3. 调用 Qwen Chat
        messages_for_llm = [
            {"role": "system", "content": st.session_state.system_prompt},
        ]
        # 把历史对话塞进去
        for m in st.session_state.chat_messages:
            if m["role"] in ("user", "assistant"):
                messages_for_llm.append(m)

        # 再附加一条带“记忆快照”的 user 消息
        messages_for_llm.append({
            "role": "user",
            "content": (
                "下面是检索到的相关记忆快照，请结合这些内容回答上面的问题：\n\n"
                + mem_ctx
            )
        })

        with st.chat_message("assistant"):
            with st.spinner("Qwen 正在思考..."):
                answer = call_qwen_chat(messages_for_llm, model_name=llm_model)
                st.markdown(answer)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )
