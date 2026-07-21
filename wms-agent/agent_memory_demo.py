#!/usr/bin/env python3
"""
Agent 长期记忆 Demo —— 基于 LangGraph Store + Checkpoint

使用 LangGraph 生态组件：
  - InMemoryStore   -> 用户长期记忆（持久化 key-value 存储）
  - SqliteSaver     -> 会话 checkpoint 持久化

演示：不同用户拥有独立的记忆空间，
      记忆可跨会话持久化，支持标签过滤和检索。
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

# -- LangGraph 组件 ------------------------------------
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver

# -- 配置 ----------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "langgraph_memory.db")

# -- 种子数据 ------------------------------------------
SEED_MEMORIES = {
    "Alice": [
        ("周末喜欢去郊外徒步，尤其钟爱山脊线", "hobby"),
        ("正在学习西班牙语，目标是年底前达到 B1 水平", "learning"),
        ("上个月收养了一只叫 Oreo 的流浪猫", "pet"),
        ("对量子计算很感兴趣，正在读 Nielsen 的教材", "interest"),
        ("最怕番茄，任何含番茄的菜都不吃", "preference"),
    ],
    "Bob": [
        ("是一名全栈工程师，擅长 React 和 Go", "career"),
        ("每天早上 6 点起床晨跑 5 公里", "routine"),
        ("正在筹备 9 月的婚礼，有点焦虑", "life_event"),
        ("不喜欢咖啡，只喝绿茶", "preference"),
        ("最近在学钢琴，已经能弹《致爱丽丝》了", "learning"),
    ],
    "Carol": [
        ("是 UX 设计师，关注可访问性设计", "career"),
        ("养了两只仓鼠，分别叫奶茶和布丁", "pet"),
        ("去年完成了一次 solo 日本旅行，最喜欢京都", "travel"),
        ("正在练习自由潜，目标深度 20 米", "learning"),
        ("是个素食主义者，但偶尔偷吃寿司", "preference"),
    ],
}

NEW_MEMORIES = {
    "Alice": [
        "今天在徒步时发现了一条新路线，拍了很多照片",
        "西班牙语老师表扬她发音进步很大",
    ],
    "Bob": [
        "今天修复了一个棘手的并发 bug，很有成就感",
    ],
    "Carol": [],
}


# -- 记忆管理器 ----------------------------------------

def _keyword_match(text: str, query: str) -> float:
    """简单关键词匹配分数，用于补偿 InMemoryStore 无 embedding 时的检索。"""
    if not query:
        return 0.0
    q = query.lower()
    t = text.lower()
    score = 0.0
    for kw in q.split():
        if kw in t:
            score += 1.0
    return score / len(q.split()) if q.split() else 0.0


class AgentMemoryStore:
    """基于 LangGraph InMemoryStore 的用户记忆管理。

    每个用户的记忆存储在 namespace ("memories", <user_id>) 下，
    每条记忆是一个 Item，key 为记忆的 uuid，value 包含文本和标签。
    """

    def __init__(self, store: InMemoryStore):
        self._store = store

    def add(
        self,
        user_id: str,
        text: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        """存储一条新记忆。"""
        key = str(uuid.uuid4())
        self._store.put(
            ("memories", user_id),
            key,
            {
                "text": text,
                "tags": tags or [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return key

    def add_with_label(self, user_id: str, text: str, label: str) -> str:
        return self.add(user_id, text, tags=[label])

    def search(
        self,
        user_id: str,
        query: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 10,
    ) -> List[dict]:
        """在用户的记忆中检索。支持按标签过滤和关键词匹配。"""
        filter_dict = {"tag": tag} if tag else None
        results = self._store.search(
            ("memories", user_id),
            query=None,  # InMemoryStore 无 index 时 query 无效
            filter=filter_dict,
            limit=100,
        )
        items = []
        for r in results:
            text = r.value.get("text", "")
            kw_score = _keyword_match(text, query or "") if query else 1.0
            items.append({
                "id": r.key,
                "text": text,
                "tags": r.value.get("tags", []),
                "score": r.score or kw_score,
                "created_at": r.value.get("created_at", ""),
            })
        if query:
            items.sort(key=lambda x: x["score"], reverse=True)
        return items[:limit]

    def get_all(self, user_id: str) -> List[dict]:
        results = self._store.search(("memories", user_id), limit=1000)
        return [
            {
                "id": r.key,
                "text": r.value.get("text", ""),
                "tags": r.value.get("tags", []),
                "created_at": r.value.get("created_at", ""),
            }
            for r in results
        ]

    def count(self, user_id: str) -> int:
        return len(self.get_all(user_id))

    def delete_all(self, user_id: str):
        for item in self.get_all(user_id):
            self._store.delete(("memories", user_id), item["id"])


# -- Checkpoint 演示 -----------------------------------

def demo_checkpoint():
    """演示 LangGraph SqliteSaver 的持久化 checkpoint。"""
    print("  [SqliteSaver] 创建 SQLite 持久化 checkpoint saver ...")
    conn = sqlite3.connect(DB_PATH)
    saver = SqliteSaver(conn)

    configs = {}
    for user_id in ["Alice", "Bob", "Carol"]:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        configs[user_id] = config
        checkpoint = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            "channel_values": {
                "user_id": user_id,
                "messages": [
                    {"role": "system", "content": "你是 %s 的 AI 助手。" % user_id},
                ],
            },
        }
        saved = saver.put(
            config, checkpoint,
            {"source": "input", "step": 1, "writes": None}, {},
        )
        cp_id = saved["configurable"]["checkpoint_id"][:8]
        print("    v %s -> checkpoint 已写入 (thread: %s, cp: %s)" % (
            user_id, thread_id[:8], cp_id))

    # Alice 多轮对话
    alice_cp = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "id": str(uuid.uuid4()),
        "channel_values": {
            "user_id": "Alice",
            "messages": [
                {"role": "system", "content": "你是 Alice 的 AI 助手。"},
                {"role": "user", "content": "我周末想去徒步，有什么推荐吗？"},
                {"role": "assistant", "content": "根据你的记忆，你钟爱山脊线路线..."},
            ],
        },
    }
    saver.put(configs["Alice"], alice_cp, {"source": "loop", "step": 2, "writes": None}, {})
    print("  [SqliteSaver] Alice 对话已更新 (共 3 条消息)")

    print("  [SqliteSaver] checkpoint 文件: %s (%s bytes)" % (
        DB_PATH, os.path.getsize(DB_PATH)))
    for user_id in ["Alice", "Bob", "Carol"]:
        tid = configs[user_id]["configurable"]["thread_id"]
        ckpts = saver.list({"configurable": {"thread_id": tid, "checkpoint_ns": ""}})
        print("    %s: %s 个 checkpoint" % (user_id, len(list(ckpts))))

    conn.close()
    return configs


# -- Demo 主函数 ---------------------------------------

def run_demo():
    print("=" * 62)
    print("  Agent 长期记忆 . LangGraph 记忆系统 Demo")
    print("  (InMemoryStore + SqliteSaver)")
    print("=" * 62)

    # 1. 初始化 Store
    print("\n[1/5] 初始化 LangGraph InMemoryStore ...")
    store = InMemoryStore()
    mem = AgentMemoryStore(store)
    print("    v InMemoryStore 就绪")

    # 2. 灌入种子记忆
    print("\n[2/5] 灌入种子记忆 ...")
    for user_id, memories in SEED_MEMORIES.items():
        for text, label in memories:
            mem.add_with_label(user_id, text, label)
        print("    v %s -> %s 条记忆" % (user_id, len(memories)))

    # 3. 记忆隔离演示：同一搜索词不同用户结果不同
    print("\n[3/5] 记忆隔离 + 语义检索")
    print("    ------------------------------------------------")
    queries = [("Alice", "徒步"), ("Bob", "咖啡"), ("Carol", "仓鼠")]
    for user_id, q in queries:
        results = mem.search(user_id, q, limit=3)
        tags_str = ", ".join(r["tags"][0] for r in results if r["tags"])
        print('\n    [%s] 搜索: "%s"' % (user_id, q))
        for r in results:
            print("      [%.3f] %s" % (r["score"], r["text"]))

    # 4. 模拟新记忆写入
    print("\n[4/5] 模拟新记忆写入 ...")
    for user_id, new_list in NEW_MEMORIES.items():
        if not new_list:
            print("    - %s 没有新记忆" % user_id)
            continue
        for text in new_list:
            mem.add(user_id, text, tags=["new"])
        print("    v %s -> 新增 %s 条, 共 %s 条" % (
            user_id, len(new_list), mem.count(user_id)))

    # 5. Checkpoint 持久化
    print("\n[5/5] Checkpoint 持久化（SqliteSaver）...")
    configs = demo_checkpoint()

    # 汇总
    print("\n" + "=" * 62)
    print("  Demo 完成 v")
    print("  记忆总数:")
    for user_id in ["Alice", "Bob", "Carol"]:
        print("    %s: %s 条记忆" % (user_id, mem.count(user_id)))
    print("  Checkpoint 文件: %s" % DB_PATH)
    print("=" * 62)


if __name__ == "__main__":
    run_demo()
