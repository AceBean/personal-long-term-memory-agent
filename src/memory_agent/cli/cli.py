import argparse
import os

from memory_agent.models.memory_index import MemoryIndexQwen


def cmd_index(args):
    idx = MemoryIndexQwen()
    idx.build_from_folder(args.root)
    idx.save(args.out)
    print(f"[OK] 索引构建完成，保存到 {args.out}")


def cmd_search(args):
    if not os.path.exists(args.index):
        raise FileNotFoundError(f"索引文件不存在：{args.index}")

    idx = MemoryIndexQwen.load(args.index)
    results = idx.search_grouped(args.query, top_k=args.top_k, max_children=args.max_children)

    print(f"=== 检索：{args.query}")
    for rank, g in enumerate(results, 1):
        parent = g["parent"]
        score = g["parent_score"]
        print(f"[{rank}] {parent.modality} | score={score:.3f}")
        print(f"    path: {parent.path}")
        print(f"    time: {parent.timestamp}")
        print(f"    preview: {parent.preview_text[:120]}")
        print()


def cmd_qa(args):
    """
    这里只做两件事：
    1）调用 search_grouped 找相关记忆
    2）调用 build_grouped_llm_context 打印出用于 LLM 的上下文
    至于要不要再调 Qwen Chat 模型，由你自己决定（更安全）
    """
    if not os.path.exists(args.index):
        raise FileNotFoundError(f"索引文件不存在：{args.index}")

    idx = MemoryIndexQwen.load(args.index)
    grouped = idx.search_grouped(args.query, top_k=args.top_k, max_children=args.max_children)
    ctx = idx.build_grouped_llm_context(args.query, grouped)

    print("========== LLM 上下文 ==========")
    print(ctx)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Qwen 多模态记忆索引 CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_index = sub.add_parser("index", help="从目录构建索引")
    p_index.add_argument("--root", required=True, help="数据根目录")
    p_index.add_argument("--out", default="memory_index.pt", help="输出索引文件路径")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = sub.add_parser("search", help="检索")
    p_search.add_argument("--index", required=True, help="已保存索引文件路径")
    p_search.add_argument("--query", required=True, help="检索文本")
    p_search.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_search.add_argument("--max-children", type=int, default=3, dest="max_children")
    p_search.set_defaults(func=cmd_search)

    # qa
    p_qa = sub.add_parser("qa", help="仅打印 RAG 上下文（不直接调 LLM）")
    p_qa.add_argument("--index", required=True)
    p_qa.add_argument("--query", required=True)
    p_qa.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_qa.add_argument("--max-children", type=int, default=3, dest="max_children")
    p_qa.set_defaults(func=cmd_qa)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
