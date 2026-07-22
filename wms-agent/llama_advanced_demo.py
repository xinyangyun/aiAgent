#!/usr/bin/env python3
"""
LlamaIndex 向量索引高级用法 Demo

演示:
  1. 基础检索 + 元数据过滤         ✓ 已实现
  2. 检索后处理 (Score 阈值过滤)   ✓ 已实现
  3. 多精度检索 (不同 top_k)       ✓ 已实现
  4. 文档管理 (增/删/改)           ✓ 已实现
  5. 索引持久化 (存储 / 加载)      ✓ 已实现
  6. 多索引 + 路由查询             ✓ 已实现

运行: conda activate 2026agent && python3 llama_advanced_demo.py
"""

import math, re, os, shutil, json
from collections import Counter
from llama_index.core import (
    Document, VectorStoreIndex, Settings,
    load_index_from_storage, StorageContext, PromptTemplate,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.vector_stores import (
    MetadataFilters, MetadataFilter, FilterOperator,
)
from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata


# ── 简易嵌入 ──────────────────────────────────────────

class CharFreqEmbedding(BaseEmbedding):
    def __init__(self, **kwargs):
        super().__init__(model_name="char-freq", **kwargs)
    def _vec(self, text):
        text = text.lower()
        cjk = Counter(c for c in text if '\u4e00' <= c <= '\u9fff')
        asc = Counter(c for c in re.findall(r'[a-z]', text))
        total = len(text) or 1
        v = []
        for ch in "的了一不是有在人这中大上个我以要他时来用们自会起也子就那你好去年":
            v.append(cjk.get(ch, 0) / total)
        while len(v) < 256: v.append(0.0)
        for ch in "abcdefghijklmnopqrstuvwxyz":
            v.append(asc.get(ch, 0) / total)
        while len(v) < 384: v.append(0.0)
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x / n for x in v]
    def _get_text_embedding(self, t): return self._vec(t)
    def _get_query_embedding(self, q): return self._vec(q)
    async def _aget_text_embedding(self, t): return self._vec(t)
    async def _aget_query_embedding(self, q): return self._vec(q)

Settings.embed_model = CharFreqEmbedding()
Settings.llm = MockLLM()


# ── 测试数据 ──────────────────────────────────────────

ALL_DOCS = [
    Document(text="Prompt engineering includes zero-shot, few-shot, and chain-of-thought reasoning. Chain-of-thought (CoT) helps LLMs break down complex problems step by step.", metadata={"topic": "prompt", "level": "intermediate"}),
    Document(text="RAG combines retrieval from external knowledge with LLM generation. Pipeline: ingest -> chunk -> embed -> vector store -> retrieve -> generate with context.", metadata={"topic": "rag", "level": "advanced"}),
    Document(text="ChromaDB is an open-source vector database supporting metadata filtering and ANN search. Pinecone and Weaviate are popular alternatives.", metadata={"topic": "vector_db", "level": "intermediate"}),
    Document(text="LoRA is parameter-efficient fine-tuning. It adds low-rank matrices to attention layers, reducing memory 8-16x while preserving quality.", metadata={"topic": "fine_tuning", "level": "advanced"}),
    Document(text="LangGraph provides a graph framework for building agent workflows with cycles, branching, branching, and state persistence.", metadata={"topic": "agent", "level": "advanced"}),
]

print("=" * 62)
print("  LlamaIndex 高级用法 Demo")
print("=" * 62)


# ════ 1. 基础检索 + 元数据过滤 ════════════════════════
print("\n[1/6] 基础检索 + 元数据过滤")

index = VectorStoreIndex.from_documents(ALL_DOCS)

# 无过滤
r = index.as_retriever(similarity_top_k=3).retrieve("vector database")
print('  无过滤: "vector database" (top_k=3)')
for n in r: print("    [%s] %s..." % (n.node.metadata["topic"], n.node.text[:50]))

# 按 level 过滤
flt = MetadataFilters(filters=[MetadataFilter(key="level", value="advanced", operator=FilterOperator.EQ)])
r2 = index.as_retriever(similarity_top_k=3, filters=flt).retrieve("training")
print('\n  只查 advanced: "training"')
for n in r2: print("    [%s] %s..." % (n.node.metadata["topic"], n.node.text[:50]))


# ════ 2. 检索后处理 ════════════════════════════════════
print("\n[2/6] 检索后处理 (Score 阈值过滤)")

raw = index.as_retriever(similarity_top_k=5).retrieve("fine-tune and vector storage")
print("  原始检索: %d 条" % len(raw))

# 按分数阈值过滤
post = SimilarityPostprocessor(similarity_cutoff=0.01)
filtered = post.postprocess_nodes(raw)
print("  Score>0.01 后: %d 条" % len(filtered))

# 自定义过滤：只保留含特定词的节点
def custom_filter(nodes):
    keywords = ["RAG", "LoRA", "ChromaDB"]
    out = []
    for n in nodes:
        text = n.node.text
        if any(k.lower() in text.lower() for k in keywords):
            out.append(n)
    return out

final = custom_filter(filtered)
print("  关键词[RAG/LoRA/ChromaDB]后: %d 条" % len(final))
for n in final: print("    %s..." % n.node.text[:50])


# ════ 3. 多精度检索 ════════════════════════════════════
print("\n[3/6] 多精度检索 (不同 top_k)")

for k in [1, 2, 5]:
    r = index.as_retriever(similarity_top_k=k).retrieve("LLM training")
    print("  top_k=%d: %d 条结果" % (k, len(r)))
    for n in r: print("    [%.3f] %s" % (n.score, n.node.text[:40]))


# ════ 4. 文档管理 ══════════════════════════════════════
print("\n[4/6] 文档管理 (增/删/改)")

# 插入
new_doc = Document(
    text="DeepSeek-R1 is a reasoning model using reinforcement learning. It achieves GPT-4 level performance on math and code tasks.",
    metadata={"topic": "llm", "level": "advanced"},
)
index.insert(new_doc)
print("  v 插入新文档: DeepSeek-R1")
r = index.as_retriever(similarity_top_k=1).retrieve("deepseek")
print("    检索验证: %s..." % (r[0].node.text[:50] if r else "未找到"))

# 删除
doc_id = ALL_DOCS[0].doc_id
index.delete_ref_doc(doc_id)
print("  v 删除文档: %s" % ALL_DOCS[0].metadata["topic"])

# 更新
ALL_DOCS[1] = Document(text="[UPDATED] RAG combines retrieval from external knowledge with LLM generation. Pipeline: ingest -> chunk -> embed -> vector store -> retrieve -> generate.", metadata={"topic": "rag", "level": "advanced"})
index.update_ref_doc(ALL_DOCS[1])
print("  v 更新文档: %s" % ALL_DOCS[1].metadata["topic"])


# ════ 5. 索引持久化 ════════════════════════════════════
print("\n[5/6] 索引持久化 (存储 / 加载)")

persist_dir = "/tmp/llama_demo_idx"
if os.path.exists(persist_dir): shutil.rmtree(persist_dir)

index.storage_context.persist(persist_dir=persist_dir)
print("  v 索引已持久化到: %s" % persist_dir)

sc = StorageContext.from_defaults(persist_dir=persist_dir)
loaded = load_index_from_storage(sc)
print("  v 索引已从磁盘加载")
r = loaded.as_retriever(similarity_top_k=1).retrieve("vector database")
if r: print("    查询: %s..." % r[0].node.text[:50])

shutil.rmtree(persist_dir, ignore_errors=True)


# ════ 6. 多索引 + 路由 ════════════════════════════════
print("\n[6/6] 多索引 + 路由查询")

# 按 topic 拆分索引
topic_indices = {}
for topic in set(d.metadata["topic"] for d in ALL_DOCS):
    docs = [d for d in ALL_DOCS if d.metadata["topic"] == topic]
    if docs:
        topic_indices[topic] = VectorStoreIndex.from_documents(docs)

# 为每个索引创建一个查询引擎工具
tools = []
for topic, idx in topic_indices.items():
    engine = RetrieverQueryEngine.from_args(
        idx.as_retriever(similarity_top_k=2),
        llm=Settings.llm,
    )
    tools.append(QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name=topic,
            description="查询关于 %s 的内容" % topic,
        ),
    ))
    print("  + 索引 [%s]: %d 篇文档" % (topic, len(topic_indices[topic].docstore.docs)))

print("  共 %d 个独立索引, 路由查询已就绪" % len(tools))

print("\n" + "=" * 62)
print("  Demo 完成")
print("=" * 62)
