"""
使用魔搭 ModelScope 模型的 RAG 系统示例

本示例展示如何使用魔搭 ModelScope 的模型进行 RAG。

运行前准备：
1. 安装 ModelScope SDK（如果还没有）:
   pip install modelscope transformers
2. 下载需要的模型（可选，会自动下载）:
   from modelscope import snapshot_download
   snapshot_download('qwen/Qwen-7B-Chat')
   snapshot_download('damo/nlp_gte_sentence-embedding_chinese-base')
3. 设置环境变量:
   export USE_MODELSCOPE="true"
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RAG.rag_system import IntelligentRAG


def main():
    # 设置使用 ModelScope
    os.environ["USE_MODELSCOPE"] = "true"
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    documents_path = project_root / "RAG" / "documents"
    
    # 初始化 RAG 系统
    # embedding_model 和 llm_model 应该是 ModelScope 模型路径
    # 常用模型：
    # - LLM: "qwen/Qwen-7B-Chat", "qwen/Qwen-14B-Chat"
    # - Embedding: "damo/nlp_gte_sentence-embedding_chinese-base"
    rag = IntelligentRAG(
        documents_path=str(documents_path),  # 文档目录路径
        embedding_model="damo/nlp_gte_sentence-embedding_chinese-base",  # ModelScope 嵌入模型
        llm_model="qwen/Qwen-7B-Chat",  # ModelScope LLM 模型
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # 构建 RAG 系统
    # k: 初始检索的文档块数量（重排序前会检索 k*2 个文档）
    # use_rerank: 是否启用重排序（需要配置本地 Reranker）
    # rerank_top_n: 重排序后保留的文档数量
    vectorstore_path = project_root / "RAG" / "vectorstore"
    rag.build(
        k=4, 
        use_rerank=False,  # ModelScope 模式下可以配置本地 Reranker
        rerank_top_n=3,
        vectorstore_path=str(vectorstore_path),  # 自动保存和加载向量存储
        force_rebuild=False  # 设置为 True 可强制重新构建
    )
    
    # 进行查询
    questions = [
        "文档的主要内容是什么？",
        "请总结一下关键信息",
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 50)
        result = rag.query(question)
        print(f"回答: {result['answer']}")
        print(f"\n参考文档数量: {len(result['source_documents'])}")
        print("=" * 50)


if __name__ == "__main__":
    main()
