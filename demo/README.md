# Demo 目录

存放各类 LLM 小 demo，每个 demo 为 workspace 子项目（独立 `pyproject.toml`）。与 hylreg_QT、hylreg_OpenCV 结构一致。

## 当前 demo

| 文件夹 | 说明 |
|--------|------|
| `vl_demo` | 视觉语言 Demo：图片 + 问题 → VL 模型回复 |
| `voice_assistant` | 语音助手（ASR → LLM → TTS，当前为文本模式） |
| `agents` | 智能体系统：LangChain Agent 封装 |
| `RAG` | RAG 系统：文档检索与生成 |

## 运行方式

进入对应子项目目录后执行：

```bash
uv sync
uv run python run_xxx.py
# 或使用入口命令（见各子项目 README）
uv run vl-demo   # 在 demo/vl_demo 下
uv run voice-assistant   # 在 demo/voice_assistant 下
```

各子项目详细说明见其目录下的 `README.md`。
