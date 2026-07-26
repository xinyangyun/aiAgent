#!/usr/bin/env python3
"""
短期记忆 — 基于 LangGraph InMemoryStore

使用 LangGraph 的 InMemoryStore 管理对话线程。
每条线程保留最近 N 轮对话，超时自动过期。
"""

import time
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass

from langgraph.store.memory import InMemoryStore


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = 0.0


class ShortMemory:
    """基于 LangGraph InMemoryStore 的短期记忆管理。

    每个 thread_id 是一条独立对话线程。
    超过 ttl 秒无访问自动过期，超 max_turns 自动裁剪。
    进程重启数据即丢失。
    """

    def __init__(self, max_turns: int = 20, ttl_seconds: float = 1800):
        self._store = InMemoryStore()
        self.max_turns = max_turns
        self.ttl = ttl_seconds

    # ── 线程管理 ────────────────────────────────────

    def create_thread(self, thread_id: Optional[str] = None) -> str:
        """创建新线程，返回 thread_id。"""
        tid = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        self._store.put(
            ("threads",),
            tid,
            {"messages": [], "created_at": time.time(), "updated_at": time.time()},
        )
        return tid

    def delete_thread(self, thread_id: str) -> None:
        self._store.delete(("threads",), thread_id)

    def list_threads(self) -> List[str]:
        """返回所有活跃（未过期）线程 ID。"""
        now = time.time()
        threads = []
        for item in self._store.search(("threads",), limit=1000):
            age = now - item.value.get("updated_at", 0)
            if age < self.ttl:
                threads.append(item.key)
        return threads

    def has_thread(self, thread_id: str) -> bool:
        item = self._store.get(("threads",), thread_id)
        if not item:
            return False
        age = time.time() - item.value.get("updated_at", 0)
        if age >= self.ttl:
            self.delete_thread(thread_id)
            return False
        return True

    # ── 消息管理 ────────────────────────────────────

    def add_message(self, thread_id: str, role: str, content: str,
                    max_turns: Optional[int] = None) -> Message:
        """添加消息，超限自动裁剪。"""
        max_turns = max_turns or self.max_turns
        item = self._store.get(("threads",), thread_id)
        data = item.value if item else {"messages": []}

        # 过期检查
        age = time.time() - data.get("updated_at", 0)
        if age >= self.ttl:
            data = {"messages": []}

        msg_obj = Message(role=role, content=content, timestamp=time.time())
        data["messages"].append({"role": role, "content": content})

        # 裁剪
        limit = max_turns * 2
        if len(data["messages"]) > limit:
            data["messages"] = data["messages"][-limit:]

        data["updated_at"] = time.time()
        if "created_at" not in data:
            data["created_at"] = time.time()

        self._store.put(("threads",), thread_id, data)
        return msg_obj

    def add_turn(self, thread_id: str, user_msg: str, assistant_msg: str,
                 max_turns: Optional[int] = None) -> None:
        self.add_message(thread_id, "user", user_msg, max_turns)
        self.add_message(thread_id, "assistant", assistant_msg, max_turns)

    def get_history(self, thread_id: str,
                    last_n: Optional[int] = None) -> List[Dict]:
        """获取历史消息 dict 列表。"""
        if not self.has_thread(thread_id):
            return []
        item = self._store.get(("threads",), thread_id)
        if not item:
            return []
        msgs = item.value.get("messages", [])
        if last_n:
            return msgs[-(last_n * 2):]
        return msgs

    def count_turns(self, thread_id: str) -> int:
        return len(self.get_history(thread_id)) // 2

    # ── 清理 ────────────────────────────────────────

    def cleanup(self) -> int:
        """清理所有过期线程，返回清理数量。"""
        now = time.time()
        count = 0
        for item in self._store.search(("threads",), limit=1000):
            age = now - item.value.get("updated_at", 0)
            if age >= self.ttl:
                self._store.delete(("threads",), item.key)
                count += 1
        return count

    def clear_all(self) -> None:
        for item in self._store.search(("threads",), limit=1000):
            self._store.delete(("threads",), item.key)


# ── 测试 ──────────────────────────────────────────────

if __name__ == "__main__":
    mem = ShortMemory(max_turns=3, ttl_seconds=1)

    tid = mem.create_thread("test-thread")
    for i in range(5):
        mem.add_turn(tid, f"消息 {i+1}", f"回答 {i+1}")

    print("=== 裁剪测试 (max_turns=3) ===")
    print(f"轮数: {mem.count_turns(tid)} (应为 3)")
    assert mem.count_turns(tid) == 3
    print("✓ 裁剪正常")

    print("\n=== 自动过期测试 (TTL=1) ===")
    import time as _t
    _t.sleep(1.5)
    expired = not mem.has_thread(tid)
    print(f"已过期: {expired}")
    assert expired
    print("✓ 自动过期正常")

    print("\n=== 过期后写入自动重建 ===")
    mem.add_turn(tid, "续写", "续答")
    print(f"轮数: {mem.count_turns(tid)} (应为 1)")
    assert mem.count_turns(tid) == 1
    print("✓ 过期后写入自动重建")

    mem2 = ShortMemory(ttl_seconds=1)
    for i in range(3):
        t = mem2.create_thread(f"t{i}")
        mem2.add_turn(t, f"Q{i}", f"A{i}")
    _t.sleep(1.5)
    n = mem2.cleanup()
    print(f"\n清理过期线程: {n} 个")
    assert n == 3
    print("✓ 批量清理正常")

    print("\n✓ 全部测试通过")
