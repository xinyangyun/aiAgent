#!/usr/bin/env python3
"""
LangGraph 简单智能体 Demo — 接入 DeepSeek API

两个 Node：
  1. query_weather  —— 查询天气（模拟工具调用）
  2. search_llm     —— 直接搜索大模型（调用 DeepSeek）

根据用户输入自动路由到对应 Node，并返回结果。
"""

import os
import sys
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ── DeepSeek 配置 ─────────────────────────────────────

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
    max_tokens=1024,
)


# ── 状态定义 ──────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: Optional[str]


# ── 天气工具（模拟） ──────────────────────────────────

WEATHER_DB = {
    "北京": "晴，18-28°C，空气质量良好",
    "上海": "多云，24-30°C，湿度较高",
    "深圳": "阵雨，26-32°C，带伞出行",
    "成都": "阴，20-26°C，适合外出",
    "广州": "雷阵雨，25-31°C",
    "杭州": "晴转多云，22-29°C",
    "武汉": "阴转小雨，19-25°C",
    "南京": "多云，20-27°C",
    "纽约": "小雨，15-22°C",
    "伦敦": "多云，12-18°C",
    "东京": "晴，22-28°C",
}


def query_weather(city: str) -> str:
    result = WEATHER_DB.get(city)
    if result:
        return f"{city}天气：{result}"
    return f"{city}天气：多云，20-28°C（模拟数据）"


# ── 路由 LLM ─────────────────────────────────────────

router_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.1,
    max_tokens=64,
)

ROUTER_SYSTEM_PROMPT = """你是一个意图分类器。判断用户的问题是"天气查询"还是"知识问答"。
- 如果用户询问天气、温度、下雨、台风等，只输出：weather
- 其他所有问题，只输出：search
只输出一个词，不要解释。"""


# ── Node 函数 ─────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """路由节点：用 LLM 判断用户意图。"""
    messages = state.get("messages", [])
    if not messages:
        return {"next_node": "search_llm"}

    last_msg = messages[-1]
    text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # LLM 分类
    response = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ])
    label = response.content.strip().lower() if hasattr(response, "content") else ""

    # 关键词兜底
    weather_kw = ["天气", "温度", "下雨", "晴天", "多云", "刮风", "台风", "降雪", "气温"]
    if label == "weather" or any(kw in text for kw in weather_kw):
        return {"next_node": "query_weather"}
    return {"next_node": "search_llm"}


def weather_agent(state: AgentState) -> AgentState:
    """天气查询 Node。"""
    messages = state.get("messages", [])
    last_msg = messages[-1]
    text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    city = "深圳"
    for c in sorted(WEATHER_DB.keys(), key=len, reverse=True):
        if c in text:
            city = c
            break

    result = query_weather(city)
    return {"messages": [AIMessage(content=f"🌤 天气助手：{result}")]}


def search_agent(state: AgentState) -> AgentState:
    """LLM 搜索 Node — 调用 DeepSeek 回答用户问题。"""
    messages = state.get("messages", [])

    llm_messages = [
        SystemMessage(content="你是一个有用的 AI 助手。请用中文回答用户的问题，语言简洁准确。"),
    ]
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage)):
            llm_messages.append(m)

    response = llm.invoke(llm_messages)
    return {"messages": [AIMessage(content=response.content)]}


# ── 条件边 ────────────────────────────────────────────

def decide_next(state: AgentState) -> Literal["query_weather", "search_llm"]:
    return state.get("next_node", "search_llm")


# ── 构建图 ────────────────────────────────────────────

def build_agent() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("query_weather", weather_agent)
    builder.add_node("search_llm", search_agent)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        decide_next,
        {
            "query_weather": "query_weather",
            "search_llm": "search_llm",
        },
    )

    builder.add_edge("query_weather", END)
    builder.add_edge("search_llm", END)

    graph = builder.compile(checkpointer=MemorySaver())
    return graph


# ── 控制台交互 ───────────────────────────────────────

def run_app():
    print("=" * 60)
    print("  LangGraph 智能体 · 控制台交互模式")
    print("  天气查询  |  知识问答（DeepSeek）")
    print("  输入 exit 或 quit 退出")
    print("=" * 60)

    agent = build_agent()
    session_id = f"session-{os.urandom(4).hex()}"
    step = 0

    while True:
        try:
            user_input = input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        text = user_input.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            print("再见！")
            break

        step += 1
        config = {"configurable": {"thread_id": f"{session_id}-{step}", "checkpoint_ns": ""}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=config,
        )
        last_msg = result["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        print(f"  {content}")


# ── 批量测试 ─────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("  LangGraph 智能体 · 批量测试")
    print("  节点: [router] -> [query_weather | search_llm] -> END")
    print("=" * 60)

    agent = build_agent()

    test_queries = [
        "深圳今天天气怎么样？",
        "什么是大语言模型？",
        "帮我查一下北京的温度",
        "用 Python 写一个快速排序",
        "明天上海会下雨吗？",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"  Q{i}: {query}")
        print(f"{'─' * 60}")

        config = {"configurable": {"thread_id": f"demo-{i}", "checkpoint_ns": ""}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        last_msg = result["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        print(f"  A: {content}\n")

    print("=" * 60)
    print("  Demo 结束")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_app()
