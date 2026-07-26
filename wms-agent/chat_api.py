#!/usr/bin/env python3
"""
FastAPI Chat API — 封装 LangGraph Agent 为 HTTP 接口

启动: uvicorn chat_api:app --reload --port 8000
文档: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from langgraph_agent_demo import build_agent
from memory.short_memory import ShortMemory
import uuid
from user.auth import login as auth_login, logout as auth_logout

# ── 应用初始化 ────────────────────────────────────────

app = FastAPI(
    title="LangGraph Agent API",
    description="天气查询 & 知识问答 · 基于 DeepSeek",
    version="1.0.0",
)

# 全局复用同一个 Agent 实例
agent = build_agent()
mem = ShortMemory(max_turns=20, ttl_seconds=1800)


# ── 数据模型 ──────────────────────────────────────────

class QueryRequest(BaseModel):
    message: str
    """用户输入的消息"""
    session_id: str = "default"
    """会话 ID，同一 session 的对话历史会累积"""


class QueryResponse(BaseModel):
    response: str
    """Agent 回复的内容"""
    session_id: str
    """当前会话 ID"""
    node: str = ""
    """实际执行的节点（query_weather / search_llm）"""


# ── 接口 ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "LangGraph Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "发送消息给 Agent，返回回复",
            "GET /health": "健康检查",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(req: LoginRequest):
    """用户登录。"""
    result = auth_login(req.username, req.password)
    if result["success"]:
        return {
            "code": 0,
            "message": "登录成功",
            "data": {"token": result["token"], "username": result["username"]},
        }
    return {"code": 1, "message": result["message"]}


@app.post("/api/logout")
def logout(req: dict):
    """退出登录。"""
    token = req.get("token", "")
    auth_logout(token)
    return {"code": 0, "message": "已退出"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """向 Agent 发送一条消息，返回回答。"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 从短期记忆加载历史（最多 10 轮）
    history = mem.get_history(req.session_id, last_n=10)

    # 构建消息：历史 + 当前问题
    msgs = []
    for h in history:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            msgs.append(AIMessage(content=h["content"]))
    msgs.append(HumanMessage(content=req.message))
    print("[AGENT] -> query：msg:%f", msgs, flush=True)

    # 每次都新建线程，历史由 ShortMemory 管理
    thread_id = f"api-{req.session_id}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    result = agent.invoke(
        {
            # "messages": msgs,
            "messages": [HumanMessage(content=req.message)],
            "intent": "",
            "need_planner": False,
            "plan": [],
            "current_step": 0,
            "phase": "",
        },
        config=config,
    )

    messages = result.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="Agent 没有返回任何消息")

    # 找最后一条有内容的 AI 消息
    content = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and len(m.content) > 5:
            content = m.content
            break
    if not content:
        content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # 保存到短期记忆
    mem.add_turn(req.session_id, req.message, content)

    return QueryResponse(
        response=content,
        session_id=req.session_id,
        node="",
    )


# ── 直接运行入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat_api:app", host="0.0.0.0", port=8000, reload=True)
