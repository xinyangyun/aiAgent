#!/usr/bin/env python3
"""
LangGraph 多能力智能体 Demo — 接入 DeepSeek API

节点:
  1. query_weather    — 查询天气（模拟工具调用）
  2. search_llm       — 直接搜索大模型（调用 DeepSeek）
  3. query_db         — 查询库存数据库（function calling）
  4. analyze_inventory— 库存数据分析 + 报告生成

流程:
  [router] → query_weather → END
           → search_llm    → END
           → query_db → analyze_inventory → END
"""

import os
import sys
import json
import uuid
from typing import Annotated, Literal, Optional, List, Dict, Any
from datetime import datetime

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage

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

router_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.1,
    max_tokens=64,
)


# ── 状态定义 ──────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: Optional[str]
    query_data: Optional[str]
    analysis_result: Optional[str]


# ══════════════════════════════════════════════════════
# 工具 1: 天气查询
# ══════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════
# 工具 2: 库存数据库（编造数据）
# ══════════════════════════════════════════════════════

INVENTORY_DB = [
    {"id": "SKU-001", "name": "无线蓝牙耳机", "category": "电子产品", "stock": 120, "sales_30d": 45, "price": 199, "warehouse": "深圳仓", "supplier": "华强电子"},
    {"id": "SKU-002", "name": "机械键盘",    "category": "电子产品", "stock": 80,  "sales_30d": 32, "price": 399, "warehouse": "深圳仓", "supplier": "键达科技"},
    {"id": "SKU-003", "name": "USB-C 数据线", "category": "配件",     "stock": 500, "sales_30d": 210, "price": 29,  "warehouse": "广州仓", "supplier": "联创线材"},
    {"id": "SKU-004", "name": "充电宝 20000mAh", "category": "电子产品", "stock": 60,  "sales_30d": 38,  "price": 149, "warehouse": "深圳仓", "supplier": "亿能电源"},
    {"id": "SKU-005", "name": "笔记本支架",  "category": "配件",     "stock": 150, "sales_30d": 25,  "price": 89,  "warehouse": "上海仓", "supplier": "锐意五金"},
    {"id": "SKU-006", "name": "鼠标垫 (大)",  "category": "配件",     "stock": 300, "sales_30d": 65,  "price": 39,  "warehouse": "广州仓", "supplier": "创美实业"},
    {"id": "SKU-007", "name": "4K 显示器 27寸", "category": "显示器",   "stock": 25,  "sales_30d": 18,  "price": 2499,"warehouse": "上海仓", "supplier": "晶彩科技"},
    {"id": "SKU-008", "name": "无线鼠标",    "category": "电子产品", "stock": 200, "sales_30d": 55,  "price": 79,  "warehouse": "深圳仓", "supplier": "罗技"},
    {"id": "SKU-009", "name": "桌面音响",    "category": "电子产品", "stock": 40,  "sales_30d": 12,  "price": 299, "warehouse": "深圳仓", "supplier": "声韵科技"},
    {"id": "SKU-010", "name": "硬盘 2TB SSD", "category": "存储",     "stock": 35,  "sales_30d": 22,  "price": 899, "warehouse": "上海仓", "supplier": "闪存先锋"},
    {"id": "SKU-011", "name": "手机壳 (通用)", "category": "配件",    "stock": 800, "sales_30d": 320, "price": 25,  "warehouse": "广州仓", "supplier": "壳乐"},
    {"id": "SKU-012", "name": "智能手表表带",  "category": "配件",    "stock": 180, "sales_30d": 48,  "price": 59,  "warehouse": "深圳仓", "supplier": "华强电子"},
    {"id": "SKU-013", "name": "摄像头 1080P",  "category": "电子产品","stock": 45,  "sales_30d": 15,  "price": 199, "warehouse": "广州仓", "supplier": "视界科技"},
    {"id": "SKU-014", "name": "路由器 WiFi6",  "category": "网络设备", "stock": 30,  "sales_30d": 20,  "price": 599, "warehouse": "上海仓", "supplier": "网盈科技"},
    {"id": "SKU-015", "name": "扩展坞 Type-C", "category": "配件",    "stock": 55,  "sales_30d": 28,  "price": 159, "warehouse": "广州仓", "supplier": "联创线材"},
]


def query_inventory(category: str = "", warehouse: str = "",
                    min_stock: int = 0, max_stock: int = 99999,
                    sort_by: str = "id", limit: int = 10) -> str:
    """查询库存数据。"""
    results = INVENTORY_DB[:]
    if category:
        results = [r for r in results if r["category"] == category]
    if warehouse:
        results = [r for r in results if warehouse in r["warehouse"]]
    results = [r for r in results if min_stock <= r["stock"] <= max_stock]
    reverse = sort_by.startswith("-")
    key = sort_by.lstrip("-")
    if key in ("stock", "sales_30d", "price"):
        results.sort(key=lambda x: x[key], reverse=reverse)
    return json.dumps(results[:limit], ensure_ascii=False, indent=2)


def get_stock_alerts() -> str:
    """库存预警：低库存<50 件 或 高库存>500 件。"""
    low = sorted([r for r in INVENTORY_DB if r["stock"] < 50], key=lambda x: x["stock"])
    high = sorted([r for r in INVENTORY_DB if r["stock"] > 500], key=lambda x: -x["stock"])
    return json.dumps({
        "低库存预警": [dict(r) for r in low],
        "高库存预警": [dict(r) for r in high],
    }, ensure_ascii=False, indent=2)


# ── 路由 LLM ─────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """你是一个意图分类器。判断用户的问题属于哪一类：
- weather  — 天气、温度、下雨、台风等天气相关
- inventory— 库存、仓库、商品、销量、预警、盘点等库存相关
- search   — 其他所有知识问答
只输出一个词，不要解释。"""


# ══════════════════════════════════════════════════════
# Node 函数
# ══════════════════════════════════════════════════════

def router_node(state: AgentState) -> AgentState:
    """路由节点：用 LLM 判断用户意图。"""
    messages = state.get("messages", [])
    if not messages:
        return {"next_node": "search_llm", "query_data": None, "analysis_result": None}

    last_msg = messages[-1]
    text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    response = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ])
    label = response.content.strip().lower() if hasattr(response, "content") else ""

    weather_kw = ["天气", "温度", "下雨", "晴天", "多云", "台风", "降雪", "气温"]
    inventory_kw = ["库存", "仓库", "商品", "销量", "预警", "盘点", "sku", "进货",
                    "缺货", "滞销", "热销", "周转", "采购", "存货"]

    if label == "weather" or any(kw in text for kw in weather_kw):
        return {"next_node": "query_weather", "query_data": None, "analysis_result": None}
    if label == "inventory" or any(kw in text for kw in inventory_kw):
        return {"next_node": "query_db", "query_data": None, "analysis_result": None}
    return {"next_node": "search_llm", "query_data": None, "analysis_result": None}


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
    """LLM 搜索 Node。"""
    messages = state.get("messages", [])
    llm_messages = [
        SystemMessage(content="你是一个有用的 AI 助手。请用中文回答，语言简洁准确。"),
    ]
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage)):
            llm_messages.append(m)

    response = llm.invoke(llm_messages)
    return {"messages": [AIMessage(content=response.content)]}


# ── Node: 库存查询 ────────────────────────────────────

INVENTORY_TOOLS = {
    "query_inventory": query_inventory,
    "get_stock_alerts": get_stock_alerts,
}

INVENTORY_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": "查询库存数据，按分类/仓库/库存量过滤",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "商品分类"},
                    "warehouse": {"type": "string", "description": "仓库名"},
                    "min_stock": {"type": "integer", "description": "最低库存"},
                    "max_stock": {"type": "integer", "description": "最高库存"},
                    "sort_by": {"type": "string", "description": "排序字段"},
                    "limit": {"type": "integer", "description": "返回条数"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_alerts",
            "description": "获取库存预警（低库存<50，高库存>500）",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def query_db_node(state: AgentState) -> AgentState:
    """通过 function calling 查询库存数据。"""
    messages = state.get("messages", [])
    if not messages:
        return {"query_data": "[]"}

    last_msg = messages[-1]
    text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    response = llm.invoke(
        [
            SystemMessage(content="你是库存查询助手。根据用户问题调用对应的工具查询数据。"),
            HumanMessage(content=text),
        ],
        tools=INVENTORY_TOOL_DEFS,
    )

    # 兼容新旧两种 function calling 响应格式
    func_call = None
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls and len(tool_calls) > 0:
        func_call = {
            "name": tool_calls[0]["name"],
            "arguments": json.dumps(tool_calls[0].get("args", {})),
        }
    elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
        tc = response.additional_kwargs.get("tool_calls")
        if tc:
            func_call = {"name": tc[0]["function"]["name"], "arguments": tc[0]["function"].get("arguments", "{}")}
        else:
            func_call = response.additional_kwargs.get("function_call")

    if func_call:
        func_name = func_call["name"]
        args = json.loads(func_call.get("arguments", "{}")) if isinstance(func_call.get("arguments"), str) else func_call.get("arguments", {})
        func = INVENTORY_TOOLS.get(func_name)
        if func:
            result = func(**args) if args else func()
            state["query_data"] = result
            state["messages"].append(FunctionMessage(content=result, name=func_name))
            return {"query_data": result}

    content = response.content if hasattr(response, "content") else str(response)
    state["query_data"] = "[]"
    state["messages"].append(AIMessage(content=content))
    return {"query_data": "[]"}


# ── Node: 库存分析 ────────────────────────────────────

def analyze_node(state: AgentState) -> AgentState:
    """分析库存数据，生成报告 + 表格。"""
    raw = state.get("query_data", "[]")
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        data = []

    if not data:
        return {
            "analysis_result": "未查询到相关数据。",
            "messages": [AIMessage(content="未查询到相关数据。")],
        }

    # 处理 dict 类型返回（如 get_stock_alerts 返回的分类结果）
    if isinstance(data, dict):
        flat = []
        for cat, items in data.items():
            for item in items:
                item["_category"] = cat
                flat.append(item)
        data = flat
 
    # 计算周转率
    for item in data:
        item["turnover_rate"] = round(item.get("sales_30d", 0) / max(item.get("stock", 0), 1), 2)

    total_stock = sum(item.get("stock", 0) for item in data)
    total_sales = sum(item.get("sales_30d", 0) for item in data)
    low_stock = [d for d in data if d.get("stock", 0) < 50]
    high_turnover = [d for d in data if d["turnover_rate"] > 1]
    slow = [d for d in data if d["turnover_rate"] < 0.3]

    # 表格
    lines = ["| 商品 | 库存 | 月销量 | 周转率 | 状态 |", "|---|---|---|---|---|"]
    for item in data:
        s = "⚠️ 低库存" if item.get("stock", 0) < 50 else ("🔥 热销" if item["turnover_rate"] > 1 else "✅ 正常")
        lines.append(f"| {item['name']} | {item['stock']} | {item['sales_30d']} | {item['turnover_rate']} | {s} |")
    table = "\n".join(lines)

    analysis = (
        f"📊 【库存分析报告】\n\n"
        f"【数据摘要】\n"
        f"- 商品数: {len(data)} 件\n"
        f"- 总库存: {total_stock} 件\n"
        f"- 月总销量: {total_sales} 件\n\n"
        f"【关键发现】\n"
    )
    if low_stock:
        analysis += f"🔴 低库存预警: {', '.join(d['name'] for d in low_stock[:3])} 等 {len(low_stock)} 件商品库存不足\n"
    if high_turnover:
        analysis += f"🔥 热销商品: {', '.join(d['name'] for d in high_turnover[:3])} 等 {len(high_turnover)} 件商品月销量高\n"
    if slow:
        analysis += f"🐢 滞销商品: {', '.join(d['name'] for d in slow[:3])} 等 {len(slow)} 件商品周转率低\n"

    suggestion_prompt = (
        f"基于以下库存数据，给出 2-3 条可操作的业务建议：\n"
        f"数据: {json.dumps(data, ensure_ascii=False)}\n"
        f"低库存: {[d['name'] for d in low_stock]}\n"
        f"热销: {[d['name'] for d in high_turnover]}\n"
        f"滞销: {[d['name'] for d in slow]}\n"
    )
    suggestion = llm.invoke([HumanMessage(content=suggestion_prompt)])
    analysis += f"\n【建议】\n{suggestion.content}"

    final_output = f"{analysis}\n\n【数据表格】\n{table}"
    return {"analysis_result": final_output, "messages": [AIMessage(content=final_output)]}


# ── 条件边 ────────────────────────────────────────────

def decide_next(state: AgentState) -> Literal["query_weather", "search_llm", "query_db"]:
    return state.get("next_node", "search_llm")


# ── 构建图 ────────────────────────────────────────────

def build_agent() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("query_weather", weather_agent)
    builder.add_node("search_llm", search_agent)
    builder.add_node("query_db", query_db_node)
    builder.add_node("analyze_inventory", analyze_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        decide_next,
        {
            "query_weather": "query_weather",
            "search_llm": "search_llm",
            "query_db": "query_db",
        },
    )

    builder.add_edge("query_weather", END)
    builder.add_edge("search_llm", END)
    builder.add_edge("query_db", "analyze_inventory")
    builder.add_edge("analyze_inventory", END)

    graph = builder.compile(checkpointer=MemorySaver())
    return graph


# ── 控制台交互 ───────────────────────────────────────

def run_app():
    print("=" * 62)
    print("  LangGraph 智能体 · 控制台交互模式")
    print("  天气查询 | 知识问答 | 库存查询与分析")
    print("  输入 exit 退出")
    print("=" * 62)

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
    print("=" * 62)
    print("  LangGraph 智能体 · 批量测试")
    print("  天气 | 知识问答 | 库存查询与数据分析")
    print("=" * 62)

    agent = build_agent()

    test_queries = [
        "深圳今天天气怎么样？",
        "什么是大语言模型？",
        "查询所有电子产品的库存情况",
        "检查库存预警信息",
        "分析深圳仓所有商品的库存",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 62}")
        print(f"  Q{i}: {query}")
        print(f"{'─' * 62}")

        config = {"configurable": {"thread_id": f"demo-{i}", "checkpoint_ns": ""}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        last_msg = result["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        print(f"\n{content}\n")

    print("=" * 62)
    print("  Demo 结束")
    print("=" * 62)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_app()
