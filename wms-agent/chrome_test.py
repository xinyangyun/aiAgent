#!/usr/bin/env python3
"""
LlamaIndex 向量存储 + 元数据 (userID) 过滤 Demo

演示:
  1. Document 携带 user_id 元数据存入向量索引
  2. 查询时通过 MetadataFilters 按 user_id 过滤
  3. 同一问题不同用户看到不同结果（记忆隔离）

运行: conda activate 2026agent && python3 chrome_test.py
"""

import math
import re
from collections import Counter
from typing import List, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import Settings


# ── 种子数据 ──────────────────────────────────────────

SEED_DATA = [
    # (文本, user_id, 标签)
    ("周末喜欢去郊外徒步，尤其钟爱山脊线", "alice", "hobby"),
    ("正在学习西班牙语，目标是年底前达到 B1 水平", "alice", "learning"),
    ("上个月收养了一只叫 Oreo 的流浪猫", "alice", "pet"),
    ("对量子计算很感兴趣，正在读 Nielsen 的教材", "alice", "interest"),
    ("最怕番茄，任何含番茄的菜都不吃", "alice", "preference"),

    ("是一名全栈工程师，擅长 React 和 Go", "bob", "career"),
    ("每天早上 6 点起床晨跑 5 公里", "bob", "routine"),
    ("正在筹备 9 月的婚礼，有点焦虑", "bob", "life_event"),
    ("不喜欢咖啡，只喝绿茶", "bob", "preference"),
    ("最近在学钢琴，已经能弹《致爱丽丝》了", "bob", "learning"),

    ("是 UX 设计师，关注可访问性设计", "carol", "career"),
    ("养了两只仓鼠，分别叫奶茶和布丁", "carol", "pet"),
    ("去年完成了一次 solo 日本旅行，最喜欢京都", "carol", "travel"),
    ("正在练习自由潜，目标深度 20 米", "carol", "learning"),
    ("是个素食主义者，但偶尔偷吃寿司", "carol", "preference"),
]


# ── 自定义嵌入模型（无需联网下载） ────────────────────

_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789" + "".join(chr(i) for i in range(0x4e00, 0x4f00))
_EMBED_DIM = 384


class CharFreqEmbedding(BaseEmbedding):
    """基于字符频率的本地嵌入模型，无需任何外部依赖。"""

    def __init__(self, **kwargs):
        super().__init__(model_name="char-freq", **kwargs)

    def _embed(self, text: str) -> List[float]:
        text = text.lower()
        # 中文字符频率
        cjk_counts = Counter(c for c in text if '\u4e00' <= c <= '\u9fff')
        # 英文字符频率
        ascii_counts = Counter(c for c in re.findall(r'[a-z]', text))
        # 总长度
        total = len(text) or 1

        vec = []
        # 前 256 维：中文高频字符
        for ch in "的了一不是有在人这中大上个我以要他时来用们自会起也子就那你好去年":
            vec.append(cjk_counts.get(ch, 0) / total)
        # 补齐到 256
        while len(vec) < 256:
            vec.append(0.0)

        # 后 128 维：英文字母频率
        for ch in "abcdefghijklmnopqrstuvwxyz":
            vec.append(ascii_counts.get(ch, 0) / total)
        while len(vec) < _EMBED_DIM:
            vec.append(0.0)

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)


# ── 构建索引 ──────────────────────────────────────────

def build_index():
    """创建 Document 并构建向量索引。"""
    docs = [
        Document(text=text, metadata={"user_id": uid, "tag": tag})
        for text, uid, tag in SEED_DATA
    ]
    # print(f"  创建 Document %f", docs)
    print(f"  创建 {len(docs)} 篇 Document ...")
    index = VectorStoreIndex.from_documents(docs)
    print("  向量索引构建完成\n")
    return index


# ── 检索工具 ──────────────────────────────────────────

def query_by_user(index, user_id: str, query: str, top_k: int = 3):
    """按 user_id 过滤后检索。"""
    filters = MetadataFilters(
        filters=[MetadataFilter(
            key="user_id", value=user_id, operator=FilterOperator.EQ,
        )]
    )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    return retriever.retrieve(query)


# ── Demo ───────────────────────────────────────────────

def run_demo():
    print("=" * 62)
    print("  LlamaIndex · 向量存储 + userID 元数据过滤")
    print("  自定义嵌入模型（零外部依赖）")
    print("=" * 62)

    # ── 1. 配置嵌入模型 ──────────────────────────────
    print("\n[1/5] 配置嵌入模型 ...")
    Settings.embed_model = CharFreqEmbedding()
    print("    v CharFreqEmbedding (384维, 字符频率)")

    # ── 2. 构建索引 ──────────────────────────────────
    print("\n[2/5] 构建向量索引 ...")
    index = build_index()

    # ── 3. 不设过滤：混合检索 ────────────────────────
    print("[3/5] 无过滤 — 所有用户混合")
    retriever_all = index.as_retriever(similarity_top_k=5)
    results_all = retriever_all.retrieve("宠物")
    print('    查询: "宠物"\n')
    for r in results_all:
        uid = r.node.metadata.get("user_id", "?")
        print("      [%s] %s" % (uid, r.node.text[:40]))
    print()

    # ── 4. 按 user_id 过滤：记忆隔离 ─────────────────
    print("[4/5] 按 user_id 过滤 — 用户隔离")
    print("    ─────────────────────────────────────────")

    test_cases = [
        ("alice", "宠物", "Alice 只看自己的记忆"),
        ("bob", "宠物", "Bob 只看自己的记忆"),
        ("carol", "旅行", "Carol 只看自己的记忆"),
    ]
    for uid, query, desc in test_cases:
        print('\n    [%s] 搜索: "%s"' % (uid, query))
        print("    (%s)" % desc)
        results = query_by_user(index, uid, query, top_k=2)
        for r in results:
            print("      [%.3f] %s" % (r.score, r.node.text[:50]))
    print()

    # ── 5. 过滤运算符演示 ────────────────────────────
    print("[5/5] 过滤运算符示例")
    print("    ─────────────────────────────────────────")

    # EQ: 精确匹配
    f_eq = MetadataFilters(
        filters=[MetadataFilter(key="user_id", value="alice", operator=FilterOperator.EQ)]
    )
    r_eq = index.as_retriever(similarity_top_k=2, filters=f_eq)
    print('\n    [EQ] alice 的 "徒步":')
    print("      %s" % r_eq.retrieve("徒步")[0].node.text[:40])

    # NE: 排除某人
    f_ne = MetadataFilters(
        filters=[MetadataFilter(key="user_id", value="alice", operator=FilterOperator.NE)]
    )
    r_ne = index.as_retriever(similarity_top_k=3, filters=f_ne)
    print('\n    [NE] 排除 alice 后查 "宠物":')
    for r in r_ne.retrieve("宠物"):
        print("      [%s] %s" % (r.node.metadata["user_id"], r.node.text[:40]))

    # IN: 多用户
    f_in = MetadataFilters(
        filters=[MetadataFilter(key="user_id", value=["alice", "bob"], operator=FilterOperator.IN)]
    )
    r_in = index.as_retriever(similarity_top_k=3, filters=f_in)
    print('\n    [IN] alice + bob 查 "学习":')
    for r in r_in.retrieve("学习"):
        print("      [%s] %s" % (r.node.metadata["user_id"], r.node.text[:40]))

    print("\n" + "=" * 62)
    print("  Demo 完成 ✓")
    print("=" * 62)


if __name__ == "__main__":
    run_demo()
