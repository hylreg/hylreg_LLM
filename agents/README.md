# 智能体系统

基于 LangChain 实现的智能体（Agent）系统，支持多种 LLM 和工具调用。

## 功能特性

- 🤖 智能体框架支持
- 🛠️ 工具调用能力
- 🔄 多轮对话支持
- 📝 记忆管理
- 🎯 任务规划与执行

## 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 环境配置

系统支持多种 LLM，按以下优先级自动选择：

| 配置方式 | 环境变量 | 说明 |
|---------|---------|------|
| **Ollama（推荐）** | `USE_OLLAMA=true` | 本地运行，无需 API Key |
| 硅基流动 API | `SILICONFLOW_API_KEY`<br>`SILICONFLOW_BASE_URL` | 需要 API Key |
| OpenAI API | `OPENAI_API_KEY` | 需要 API Key |

### 使用 Ollama（推荐）

```bash
# 1. 安装 Ollama（如果还没有）
# 访问 https://ollama.ai 下载安装

# 2. 下载需要的模型
ollama pull qwen3:0.6b

# 3. 设置环境变量
export USE_OLLAMA="true"
# 或在 .env 文件中配置：
# USE_OLLAMA=true
# OLLAMA_BASE_URL=http://localhost:11434  # 默认地址，可省略
```

### 使用 API 服务

```bash
# 硅基流动
export SILICONFLOW_API_KEY="your-api-key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1/"

# 或 OpenAI
export OPENAI_API_KEY="your-api-key"
```

## 快速开始

```bash
# 从项目根目录运行
cd /Users/admin/Projects/hylreg_LLM

# 使用 Ollama（推荐）
export USE_OLLAMA="true"
uv run python agents/example.py
```

## 使用方法

### 基本使用

```python
from agent_system import IntelligentAgent

# 初始化智能体
agent = IntelligentAgent(
    llm_model="qwen3:0.6b",
    tools=[],  # 添加你的工具
)

# 运行智能体
response = agent.run("你的问题或任务")
print(response)
```

## 文件说明

- `agent_system.py`: 智能体系统核心实现
- `example.py`: 使用示例
- `tools/`: 工具目录（可选）

## 常见问题

### 1. Ollama 连接失败

```bash
# 检查 Ollama 服务是否运行
curl http://localhost:11434/api/tags

# 如果失败，启动 Ollama 服务
ollama serve

# 检查环境变量
echo $USE_OLLAMA
```

### 2. 模型未找到

```bash
# 检查已下载的模型
ollama list

# 下载缺失的模型
ollama pull qwen3:0.6b
```

## 注意事项

- **Ollama**: 需要先安装并运行 Ollama 服务，下载相应的模型
- **API Key**: 使用硅基流动或 OpenAI 时需要有效的 API Key
- 系统会自动检测环境变量，优先级：Ollama > 硅基流动 > OpenAI

## 更多文档

- [项目主文档](../README.md) - 项目概览和快速开始

