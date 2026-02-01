# 语音助手 Demo

实现「语音输入 → 大模型理解与回复 → 语音输出」的闭环；当前为**文本模式**（终端输入/输出）。

方案见 [语音助手方案.md](./语音助手方案.md)。

## 目录结构

```
voice_assistant/              # 本子项目目录，包内按功能分子目录
├── README.md
├── 语音助手方案.md
├── run_voice_assistant.py    # 入口
├── __init__.py
├── llm/                      # LLM 客户端
│   ├── __init__.py
│   └── client.py
├── audio/                    # 语音输入输出（ASR / TTS）
│   ├── __init__.py
│   ├── asr.py
│   └── tts.py
└── core/                     # 流水线：ASR → LLM → TTS
    ├── __init__.py
    └── pipeline.py
```

## 运行方式（文本模式）

**推荐**：在**本目录**下执行：

```bash
uv sync
uv run python run_voice_assistant.py
# 或
uv run voice-assistant
```

若在仓库根目录：`uv sync` 后执行 `uv run --package voice-assistant voice-assistant`。

按提示输入一句话回车，助手会调用 LLM 并打印回复；输入空行退出。

## 环境变量

与主项目一致（如从 `~/.zshrc` 读取）：

- `USE_OLLAMA=true`：使用本地 Ollama
- `OLLAMA_BASE_URL`：默认 `http://localhost:11434`
- `SILICONFLOW_API_KEY`：硅基流动 API（可选）
- `USE_MODELSCOPE`：ModelScope 本地管道（可选）

## 后续

- 阶段二：在 `audio/asr.py`、`audio/tts.py` 中接入 ModelScope FunASR / TTS，实现完整语音闭环。
