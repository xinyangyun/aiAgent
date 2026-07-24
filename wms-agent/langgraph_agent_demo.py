#!/usr/bin/env python3
"""
LangGraph Agent — 分层架构

架构:
  [Intent Classifier] → [Router] ──→ [Chat Node]     → END
                                  ├─→ [Weather Node]  → END
                                  ├─→ [Inventory Node] → END
                                  └─→ [Planner] → [Executor] ⇄ [Tools] → END

设计原则:
  - Graph 负责 Workflow，Node 代表 Capability，Tool 负责执行
  - 新增 Tool 不修改 Graph，新增 Capability 才加 Node
  - Intent Classifier 分析意图，Router 纯控制，Planner 只规划，Executor 只执行
"""

import os
import sys
import json
from typing import Annotated, Literal, Optional, List

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, temperature=0.3, max_tokens=2048)
fast_llm = ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, temperature=0.1, max_tokens=128)


# ══════════════════════════════════════════════════════
# 工具定义
# ══════════════════════════════════════════════════════

WEATHER_DB = {
    "北京": "晴，18-28°C", "上海": "多云，24-30°C", "深圳": "阵雨，26-32°C",
    "成都": "阴，20-26°C", "广州": "雷阵雨，25-31°C", "杭州": "晴转多云，22-29°C",
    "武汉": "阴转小雨，19-25°C", "南京": "多云，20-27°C",
    "纽约": "小雨，15-22°C", "伦敦": "多云，12-18°C", "东京": "晴，22-28°C",
}

INVENTORY_DB = [
    {"id": "SKU-001", "name": "无线蓝牙耳机", "category": "电子产品", "stock": 120, "sales_30d": 45, "price": 199, "warehouse": "深圳仓"},
    {"id": "SKU-002", "name": "机械键盘",    "category": "电子产品", "stock": 80,  "sales_30d": 32, "price": 399, "warehouse": "深圳仓"},
    {"id": "SKU-003", "name": "USB-C 数据线", "category": "配件",     "stock": 500, "sales_30d": 210, "price": 29,  "warehouse": "广州仓"},
    {"id": "SKU-004", "name": "充电宝 20000mAh", "category": "电子产品", "stock": 60,  "sales_30d": 38,  "price": 149, "warehouse": "深圳仓"},
    {"id": "SKU-005", "name": "笔记本支架",  "category": "配件",     "stock": 150, "sales_30d": 25,  "price": 89,  "warehouse": "上海仓"},
    {"id": "SKU-006", "name": "鼠标垫 (大)",  "category": "配件",     "stock": 300, "sales_30d": 65,  "price": 39,  "warehouse": "广州仓"},
    {"id": "SKU-007", "name": "4K 显示器 27寸", "category": "显示器",   "stock": 25,  "sales_30d": 18,  "price": 2499,"warehouse": "上海仓"},
    {"id": "SKU-008", "name": "无线鼠标",    "category": "电子产品", "stock": 200, "sales_30d": 55,  "price": 79,  "warehouse": "深圳仓"},
    {"id": "SKU-009", "name": "桌面音响",    "category": "电子产品", "stock": 40,  "sales_30d": 12,  "price": 299, "warehouse": "深圳仓"},
    {"id": "SKU-010", "name": "硬盘 2TB SSD", "category": "存储",     "stock": 35,  "sales_30d": 22,  "price": 899, "warehouse": "上海仓"},
]

def query_weather(city: str) -> str:
    r = WEATHER_DB.get(city)
    return json.dumps({"city": city, "weather": r or "多云，20-28°C（模拟）"}, ensure_ascii=False)

def query_inventory(category: str = "", warehouse: str = "", min_stock: int = 0, max_stock: int = 99999, sort_by: str = "id", limit: int = 10) -> str:
    results = [r for r in INVENTORY_DB
               if (not category or r["category"] == category)
               and (not warehouse or warehouse in r["warehouse"])
               and min_stock <= r["stock"] <= max_stock]
    key = sort_by.lstrip("-")
    if key in ("stock", "sales_30d", "price"):
        results.sort(key=lambda x: x[key], reverse=sort_by.startswith("-"))
    return json.dumps(results[:limit], ensure_ascii=False)

def get_stock_alerts() -> str:
    low = sorted([r for r in INVENTORY_DB if r["stock"] < 50], key=lambda x: x["stock"])
    high = sorted([r for r in INVENTORY_DB if r["stock"] > 500], key=lambda x: -x["stock"])
    return json.dumps({"低库存预警": [dict(r) for r in low], "高库存预警": [dict(r) for r in high]}, ensure_ascii=False)


ALL_TOOLS = {"query_weather": query_weather, "query_inventory": query_inventory, "get_stock_alerts": get_stock_alerts}

ALL_TOOL_DEFS = [
    {"type": "function", "function": {"name": "query_weather", "description": "查天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "query_inventory", "description": "查库存，支持分类/仓库/库存量过滤",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string"}, "warehouse": {"type": "string"},
            "min_stock": {"type": "integer"}, "max_stock": {"type": "integer"},
            "sort_by": {"type": "string"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "get_stock_alerts", "description": "库存预警（低<50，高>500）",
        "parameters": {"type": "object", "properties": {}}}},
]

WEATHER_TOOLS = {"query_weather": query_weather}
WEATHER_TOOL_DEFS = [ALL_TOOL_DEFS[0]]

INVENTORY_TOOLS = {"query_inventory": query_inventory, "get_stock_alerts": get_stock_alerts}
INVENTORY_TOOL_DEFS = [ALL_TOOL_DEFS[1], ALL_TOOL_DEFS[2]]


# ══════════════════════════════════════════════════════
# 状态
# ══════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str                 # "chat" | "weather" | "inventory" | "complex"
    need_planner: bool          # 是否需要规划
    plan: Optional[List[str]]   # 步骤计划
    current_step: int           # 当前执行到第几步
    phase: str                  # 流程阶段


# ══════════════════════════════════════════════════════
# Phase 1: Intent Classifier
# ══════════════════════════════════════════════════════

INTENT_PROMPT = """分析用户意图，输出 JSON：
{
  "intent": "chat | weather | inventory | complex",
  "need_planner": true | false
}

规则:
- weather: 天气、温度、下雨等
- inventory: 库存、仓库、商品、销量等
- chat: 其他简单问答（你是谁、什么是XX等）
- complex: 包含多个子任务或需要多步操作
- need_planner=true: 仅当 intent=complex 或明显需要多步骤时
"""

def intent_classifier_node(state: AgentState) -> AgentState:
    print("[AGENT] -> intent_classifier_node", flush=True)
    messages = state.get("messages", [])
    user_text = ""
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            user_text = m.content
            break

    r = fast_llm.invoke([
        SystemMessage(content=INTENT_PROMPT + f"\n用户: {user_text}"),
        HumanMessage(content="输出 JSON"),
    ])
    try:
        data = json.loads(r.content.strip().strip("```json").strip("```").strip())
        intent = data.get("intent", "chat")
        need_planner = data.get("need_planner", False)
    except Exception:
        intent, need_planner = "chat", False

    # 关键词兜底：含对比/比较等 → 强制走规划
    complex_kw = ["对比", "比较", "分别", "汇总", "总结", "所有"]
    if any(kw in user_text for kw in complex_kw):
        intent = "complex"
        need_planner = True
        print(f"[AGENT] 关键词命中 complex, 强制规划", flush=True)

    # 根据 intent 设置 phase
    if need_planner:
        phase = "plan"
    else:
        phase = intent  # "chat" | "weather" | "inventory"

    import logging
    logger = logging.getLogger("agent")
    logger.info(f"[INTENT] intent={intent}, need_planner={need_planner}, phase={phase}")
    print(f"[AGENT] intent={intent}, need_planner={need_planner}", flush=True)
    return {"intent": intent, "need_planner": need_planner, "phase": phase}


# ══════════════════════════════════════════════════════
# Phase 2: Capability Nodes
# ══════════════════════════════════════════════════════

def _call_with_tools(state, system_prompt: str, tool_map: dict, tool_defs: list) -> AgentState:
    """通用：调用 LLM + 可选工具执行。"""
    messages = state.get("messages", [])

    llm_msgs = [SystemMessage(content=system_prompt)]
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            llm_msgs.append(m)

    response = llm.invoke(llm_msgs, tools=tool_defs)

    # 检查是否有 tool_calls
    tcs = getattr(response, "tool_calls", None)
    if not tcs and hasattr(response, "additional_kwargs"):
        raw = response.additional_kwargs.get("tool_calls")
        if raw:
            tcs = [{"id": t.get("id", f"call_{i}"), "name": t["function"]["name"], "args": json.loads(t["function"].get("arguments", "{}"))} for i, t in enumerate(raw)]

    if tcs:
        tool_msgs = []
        for tc in tcs:
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            tid = tc.get("id", f"call_{hash(name)}")
            if hasattr(tc, "id"):
                tid = tc.id
            elif isinstance(tc, dict):
                tid = tc.get("id", f"call_{hash(name)}")
            raw_args = tc.get("args") or {}
            if hasattr(tc, "args"):
                raw_args = tc.args
            args = raw_args if isinstance(raw_args, dict) else {}
            func = tool_map.get(name)
            if func:
                output = func(**{k: v for k, v in args.items() if v != "" and v is not None})
                tool_msgs.append(ToolMessage(content=output, tool_call_id=tid, name=name))

        final = llm.invoke(llm_msgs + [response] + tool_msgs)
        return {"messages": [response] + tool_msgs + [final]}

    return {"messages": [response]}


def chat_node(state: AgentState) -> AgentState:
    return _call_with_tools(state, "你是一个有用的 AI 助手。请用中文回答，简洁准确。", {}, [])


def weather_node(state: AgentState) -> AgentState:
    return _call_with_tools(
        state,
        "你是一个天气查询助手。如需查天气请调用 query_weather 工具。",
        WEATHER_TOOLS, WEATHER_TOOL_DEFS,
    )


def inventory_node(state: AgentState) -> AgentState:
    return _call_with_tools(
        state,
        "你是一个库存管理助手。可调用 query_inventory 查库存数据，get_stock_alerts 查预警。拿到数据后分析周转率、标注低库存/热销状态，输出报告表格。",
        INVENTORY_TOOLS, INVENTORY_TOOL_DEFS,
    )


# ══════════════════════════════════════════════════════
# Phase 3: Planner + Executor（复杂任务）


def capability_node(state: AgentState) -> AgentState:
    print("[AGENT] -> capability_node", flush=True)
    """通用能力节点：LLM 自己判断需要调用什么工具。"""
    return _call_with_tools(
        state,
        "你是一个 AI 助手，可以使用以下工具。根据问题选择调用合适的工具，拿到数据后分析并回答。",
        ALL_TOOLS, ALL_TOOL_DEFS,
    )



# ══════════════════════════════════════════════════════
# Phase 3: Planner + Executor（复杂任务）
# ══════════════════════════════════════════════════════

PLANNER_PROMPT = """将用户任务拆解为有序执行步骤。每步独立，最后一步是汇总输出。

用户: {query}

格式（每行一个步骤，加序号）:
1. 第一步
2. 第二步"""


def planner_node(state: AgentState) -> AgentState:
    print("[AGENT] -> planner_node", flush=True)
    messages = state.get("messages", [])
    user_text = ""
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            user_text = m.content
            break

    r = llm.invoke([SystemMessage(content=PLANNER_PROMPT.format(query=user_text)), HumanMessage(content="制定计划")])
    lines = [l.strip() for l in r.content.split("\n") if l.strip() and l[0].isdigit()]
    steps = [l.split(".", 1)[1].strip() for l in lines if "." in l]
    if not steps:
        steps = [user_text]

    msg = "📋 计划:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    print("[AGENT] -> planner_node 步骤：%f", msg, flush=True)
    return {"plan": steps, "current_step": 0, "phase": "exec_plan", "messages": [AIMessage(content=msg)]}


EXEC_PROMPT = """执行以下步骤 ({current}/{total}): {step}

可调用工具获取数据，拿到数据后分析并输出结果。"""


def executor_node(state: AgentState) -> AgentState:
    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)
    messages = state.get("messages", [])

    if step_idx >= len(plan):
        return {"phase": "done"}

    prompt = EXEC_PROMPT.format(current=step_idx + 1, total=len(plan), step=plan[step_idx])

    llm_msgs = [SystemMessage(content=prompt)]
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            llm_msgs.append(m)

    response = llm.invoke(llm_msgs, tools=ALL_TOOL_DEFS)

    # 检查工具调用（在 Node 内部处理，不经过 Graph）
    tcs = getattr(response, "tool_calls", None)
    if not tcs and hasattr(response, "additional_kwargs"):
        raw = response.additional_kwargs.get("tool_calls")
        if raw:
            tcs = [{"id": t.get("id", f"call_{i}"), "name": t["function"]["name"], "args": json.loads(t["function"].get("arguments", "{}"))} for i, t in enumerate(raw)]

    if tcs:
        tool_msgs = []
        for tc in tcs:
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            tid = tc.get("id", f"call_{hash(name)}")
            if hasattr(tc, "id"):
                tid = tc.id
            elif isinstance(tc, dict):
                tid = tc.get("id", f"call_{hash(name)}")
            args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
            if hasattr(tc, "args"):
                args = tc.args
            func = ALL_TOOLS.get(name)
            if func:
                output = func(**{k: v for k, v in args.items() if v != "" and v is not None})
                tool_msgs.append(ToolMessage(content=output, tool_call_id=tid, name=name))
        final = llm.invoke(llm_msgs + [response] + tool_msgs)
        next_step = step_idx + 1
        phase = "exec_plan" if next_step < len(plan) else "done"
        print(f"[EXEC] 步骤 {step_idx+1}/{len(plan)} 完成 (有工具调用), 推进到 next_step={next_step}, phase={phase}", flush=True)
        return {"messages": [response] + tool_msgs + [final], "current_step": next_step, "phase": phase}

    next_step = step_idx + 1
    phase = "exec_plan" if next_step < len(plan) else "done"
    print(f"[EXEC] 步骤 {step_idx+1}/{len(plan)} 完成 (无工具调用), 推进到 next_step={next_step}, phase={phase}", flush=True)
    return {"messages": [response], "current_step": next_step, "phase": phase}



# ══════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════

def route_after_classify(state: AgentState) -> str:
    """Router: 根据 Intent Classifier 的结果路由，不调 LLM。"""
    need_p = state.get("need_planner", False)
    if need_p:
        return "planner"
    return "capability"


def route_after_capability(state: AgentState) -> Literal["__end__"]:
    """能力节点执行完后结束。"""
    return "__end__"


# ══════════════════════════════════════════════════════
# 构建
# ══════════════════════════════════════════════════════

def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("capability", capability_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)

    builder.set_entry_point("intent_classifier")

    # classify → Router（纯控制，不调 LLM）
    builder.add_conditional_edges(
        "intent_classifier", route_after_classify,
        {"capability": "capability", "planner": "planner"},
    )

    # Capability Nodes → END
    builder.add_edge("capability", END)

    # Planner → Executor
    builder.add_edge("planner", "executor")

    # Executor → 根据 phase 路由（步骤由内部自动推进）
    builder.add_conditional_edges(
        "executor",
        lambda s: "executor" if s.get("phase") == "exec_plan" else "__end__",
        {"executor": "executor", "__end__": END},
    )


    return builder.compile(checkpointer=MemorySaver())


# ── 交互 ──────────────────────────────────────────────

def run_app():
    print("=" * 62)
    print("  LangGraph Agent · 分层架构")
    print("  Intent Classifier → Router → Capability Node / Planner")
    print("  输入 exit 退出")
    print("=" * 62)
    agent = build_agent()
    sid = f"sess-{os.urandom(4).hex()}"
    step = 0
    while True:
        try:
            text = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！"); break
        if not text: continue
        if text.lower() in ("exit", "quit"): print("再见！"); break
        step += 1
        config = {"configurable": {"thread_id": f"{sid}-{step}", "checkpoint_ns": ""}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=text)], "intent": "", "need_planner": False,
             "plan": [], "current_step": 0, "phase": ""},
            config=config,
        )
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage) and m.content and len(m.content) > 5:
                print(f"\n  {m.content}")
                break


if __name__ == "__main__":
    run_app()
