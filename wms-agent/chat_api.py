#!/usr/bin/env python3
"""
FastAPI Chat API — 封装 LangGraph Agent 为 HTTP 接口

启动: uvicorn chat_api:app --reload --port 8000
文档: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from langgraph_agent_demo import build_agent

# ── 应用初始化 ────────────────────────────────────────

app = FastAPI(
    title="LangGraph Agent API",
    description="天气查询 & 知识问答 · 基于 DeepSeek",
    version="1.0.0",
)

# 全局复用同一个 Agent 实例
agent = build_agent()


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


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """向 Agent 发送一条消息，返回回答。"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    thread_id = f"api-{req.session_id}"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    messages = result.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="Agent 没有返回任何消息")

    last_msg = messages[-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # 判断执行节点
    node = result.get("next_node", "")

    return QueryResponse(
        response=content,
        session_id=req.session_id,
        node=node,
    )


# ── 直接运行入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat_api:app", host="0.0.0.0", port=8000, reload=True)
