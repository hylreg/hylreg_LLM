"""
RAG 系统性能测试脚本

本脚本用于测试和统计 RAG 系统的各项性能指标，包括：
- 文档加载时间
- 文档分割时间
- 向量化时间
- 向量存储创建时间
- 查询响应时间（检索、重排序、LLM生成）
- 内存使用情况
- 吞吐量测试

使用方法：
    # 使用 uv（推荐）
    uv run python benchmark.py
    
    # 或从项目根目录运行
    uv run python RAG/benchmark.py
    
    # 使用传统 Python
    python benchmark.py

环境变量：
    USE_OLLAMA=true              # 使用 Ollama 后端
    OLLAMA_BASE_URL=http://localhost:11434
    SILICONFLOW_API_KEY=...      # 使用硅基流动 API
    COHERE_API_KEY=...           # 使用 Cohere Reranker
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from rag_system import IntelligentRAG

# 尝试导入 psutil，如果不可用则使用简化版本
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil 未安装，内存监控功能将不可用。安装命令: pip install psutil")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        else:
            self.start_memory = 0
        
    def record(self, name: str, value: float, unit: str = "秒"):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "value": value,
            "unit": unit,
            "timestamp": time.time()
        })
        
    def get_current_memory(self) -> float:
        """获取当前内存使用（MB）"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        else:
            return 0.0
    
    def get_memory_delta(self) -> float:
        """获取内存增量（MB）"""
        if self.start_memory is None:
            return 0
        return self.get_current_memory() - self.start_memory
    
    def get_total_time(self) -> float:
        """获取总耗时"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        summary = {
            "总耗时": f"{self.get_total_time():.2f} 秒",
            "内存使用": f"{self.get_current_memory():.2f} MB",
            "内存增量": f"{self.get_memory_delta():.2f} MB",
            "指标详情": {}
        }
        
        for name, records in self.metrics.items():
            if records:
                values = [r["value"] for r in records]
                summary["指标详情"][name] = {
                    "平均值": f"{sum(values) / len(values):.3f} {records[0]['unit']}",
                    "最小值": f"{min(values):.3f} {records[0]['unit']}",
                    "最大值": f"{max(values):.3f} {records[0]['unit']}",
                    "总计": f"{sum(values):.3f} {records[0]['unit']}",
                    "次数": len(values)
                }
        
        return summary


class RAGPerformanceBenchmark:
    """RAG 性能测试类"""
    
    def __init__(self, rag: IntelligentRAG):
        self.rag = rag
        self.monitor = PerformanceMonitor()
        
    def benchmark_build(self, k: int = 4, use_rerank: bool = True, 
                       rerank_top_n: int = 3, vectorstore_path: Optional[str] = None,
                       force_rebuild: bool = False) -> Dict:
        """测试构建性能"""
        print("\n" + "="*60)
        print("开始测试 RAG 系统构建性能...")
        print("="*60)
        
        self.monitor.start()
        
        # 测试文档加载
        start = time.time()
        documents = self.rag.load_documents()
        load_time = time.time() - start
        self.monitor.record("文档加载时间", load_time)
        print(f"✓ 文档加载完成: {load_time:.3f} 秒 ({len(documents)} 个文档)")
        
        # 统计文档信息
        total_chars = sum(len(doc.page_content) for doc in documents)
        self.monitor.record("文档总数", len(documents), "个")
        self.monitor.record("文档总字符数", total_chars, "字符")
        
        # 测试文档分割
        start = time.time()
        splits = self.rag.split_documents(documents)
        split_time = time.time() - start
        self.monitor.record("文档分割时间", split_time)
        print(f"✓ 文档分割完成: {split_time:.3f} 秒 ({len(splits)} 个块)")
        
        # 统计块信息
        avg_chunk_size = sum(len(split.page_content) for split in splits) / len(splits) if splits else 0
        self.monitor.record("文档块总数", len(splits), "个")
        self.monitor.record("平均块大小", avg_chunk_size, "字符")
        
        # 测试向量化（创建向量存储）
        start = time.time()
        self.rag.create_vectorstore(splits)
        vectorize_time = time.time() - start
        self.monitor.record("向量化时间", vectorize_time)
        print(f"✓ 向量化完成: {vectorize_time:.3f} 秒")
        
        # 计算向量化速度
        if splits:
            vectors_per_sec = len(splits) / vectorize_time if vectorize_time > 0 else 0
            self.monitor.record("向量化速度", vectors_per_sec, "块/秒")
        
        # 测试保存向量存储
        if vectorstore_path:
            start = time.time()
            try:
                self.rag.save_vectorstore(vectorstore_path)
                save_time = time.time() - start
                self.monitor.record("向量存储保存时间", save_time)
                print(f"✓ 向量存储保存完成: {save_time:.3f} 秒")
            except Exception as e:
                print(f"✗ 向量存储保存失败: {e}")
        
        # 测试创建问答链
        start = time.time()
        self.rag.create_qa_chain(k=k, use_rerank=use_rerank, rerank_top_n=rerank_top_n)
        chain_time = time.time() - start
        self.monitor.record("问答链创建时间", chain_time)
        print(f"✓ 问答链创建完成: {chain_time:.3f} 秒")
        
        # 记录内存使用
        memory_usage = self.monitor.get_current_memory()
        memory_delta = self.monitor.get_memory_delta()
        self.monitor.record("构建后内存使用", memory_usage, "MB")
        self.monitor.record("构建内存增量", memory_delta, "MB")
        
        total_time = self.monitor.get_total_time()
        print(f"\n✓ 构建总耗时: {total_time:.3f} 秒")
        print(f"✓ 内存使用: {memory_usage:.2f} MB (增量: {memory_delta:.2f} MB)")
        
        return self.monitor.get_summary()
    
    def benchmark_query(self, questions: List[str], warmup: int = 1) -> Dict:
        """测试查询性能"""
        print("\n" + "="*60)
        print("开始测试查询性能...")
        print("="*60)
        
        if not questions:
            print("✗ 没有提供测试问题")
            return {}
        
        # 预热（第一次查询通常较慢）
        if warmup > 0:
            print(f"预热中（{warmup} 次查询）...")
            for i in range(warmup):
                try:
                    self.rag.query(questions[0] if questions else "测试")
                except:
                    pass
        
        # 测试查询
        query_times = []
        retrieval_times = []
        rerank_times = []
        llm_times = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}/{len(questions)}: {question[:50]}...")
            
            # 测量总查询时间
            start = time.time()
            result = self.rag.query(question)
            total_time = time.time() - start
            query_times.append(total_time)
            
            # 记录源文档数量
            num_docs = len(result.get("source_documents", []))
            self.monitor.record("检索文档数量", num_docs, "个")
            
            print(f"  ✓ 查询完成: {total_time:.3f} 秒 (检索到 {num_docs} 个文档)")
            print(f"  ✓ 答案长度: {len(result.get('answer', ''))} 字符")
        
        # 计算统计信息
        if query_times:
            avg_time = sum(query_times) / len(query_times)
            min_time = min(query_times)
            max_time = max(query_times)
            
            self.monitor.record("查询总时间", sum(query_times), "秒")
            self.monitor.record("平均查询时间", avg_time, "秒")
            self.monitor.record("最快查询时间", min_time, "秒")
            self.monitor.record("最慢查询时间", max_time, "秒")
            
            # 计算吞吐量
            qps = len(query_times) / sum(query_times) if sum(query_times) > 0 else 0
            self.monitor.record("查询吞吐量", qps, "查询/秒")
            
            print(f"\n✓ 平均查询时间: {avg_time:.3f} 秒")
            print(f"✓ 最快查询: {min_time:.3f} 秒")
            print(f"✓ 最慢查询: {max_time:.3f} 秒")
            print(f"✓ 查询吞吐量: {qps:.2f} 查询/秒")
        
        return self.monitor.get_summary()
    
    def benchmark_load_vectorstore(self, vectorstore_path: str) -> Dict:
        """测试加载向量存储性能"""
        print("\n" + "="*60)
        print("开始测试向量存储加载性能...")
        print("="*60)
        
        self.monitor.start()
        
        start = time.time()
        self.rag.load_vectorstore(vectorstore_path)
        load_time = time.time() - start
        self.monitor.record("向量存储加载时间", load_time)
        
        memory_usage = self.monitor.get_current_memory()
        self.monitor.record("加载后内存使用", memory_usage, "MB")
        
        print(f"✓ 向量存储加载完成: {load_time:.3f} 秒")
        print(f"✓ 内存使用: {memory_usage:.2f} MB")
        
        return self.monitor.get_summary()
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """生成性能报告"""
        summary = self.monitor.get_summary()
        
        report = {
            "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "系统配置": {
                "嵌入模型": self.rag.embedding_model,
                "LLM模型": self.rag.llm_model,
                "文档路径": self.rag.documents_path,
                "块大小": self.rag.chunk_size,
                "块重叠": self.rag.chunk_overlap,
            },
            "性能指标": summary
        }
        
        # 打印报告
        print("\n" + "="*60)
        print("性能测试报告")
        print("="*60)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 报告已保存到: {output_file}")
        
        return report


def main():
    """主测试函数"""
    import os
    from pathlib import Path
    
    # 设置使用 Ollama（根据实际情况调整）
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    if not use_ollama:
        print("提示: 设置 USE_OLLAMA=true 环境变量可使用 Ollama 后端")
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    documents_path = script_dir / "documents"
    vectorstore_path = script_dir / "vectorstore"
    
    # 初始化 RAG 系统
    print("初始化 RAG 系统...")
    rag = IntelligentRAG(
        documents_path=str(documents_path),
        embedding_model="qwen3-embedding:0.6b",
        llm_model="qwen3:0.6b",
        chunk_size=1000,
        chunk_overlap=200,
        ollama_reranker_model="dengcao/Qwen3-Reranker-0.6B:Q8_0" if use_ollama else None,
    )
    
    # 创建性能测试实例
    benchmark = RAGPerformanceBenchmark(rag)
    
    # 测试1: 构建性能（强制重建）
    print("\n【测试1】构建性能测试")
    build_summary = benchmark.benchmark_build(
        k=4,
        use_rerank=True,
        rerank_top_n=3,
        vectorstore_path=str(vectorstore_path),
        force_rebuild=True  # 强制重建以测试完整流程
    )
    
    # 测试2: 加载向量存储性能
    print("\n【测试2】向量存储加载性能测试")
    load_summary = benchmark.benchmark_load_vectorstore(str(vectorstore_path))
    
    # 测试3: 查询性能
    print("\n【测试3】查询性能测试")
    test_questions = [
        "文档的主要内容是什么？",
        "请总结一下关键信息",
        "RAG 系统的主要优势有哪些？",
        "如何使用这个系统？",
    ]
    query_summary = benchmark.benchmark_query(test_questions, warmup=1)
    
    # 生成完整报告
    report_file = script_dir / "performance_report.json"
    benchmark.generate_report(str(report_file))
    
    print("\n" + "="*60)
    print("性能测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
