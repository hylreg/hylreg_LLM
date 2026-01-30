"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
智能体系统使用示例
"""

from agents.agent_system import IntelligentAgent


def main():
    """主函数"""
    print("=" * 50)
    print("智能体系统示例")
    print("=" * 50)
    
    # 初始化智能体（不使用工具）
    agent = IntelligentAgent(
        llm_model="qwen3:0.6b",
        verbose=True,
    )
    
    # 示例查询
    queries = [
        "你好，请介绍一下你自己",
        "什么是人工智能？",
        "请帮我写一个Python函数来计算斐波那契数列",
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        print("-" * 50)
        result = agent.run(query)
        print(f"智能体: {result['answer']}")
        print("=" * 50)


if __name__ == "__main__":
    main()

