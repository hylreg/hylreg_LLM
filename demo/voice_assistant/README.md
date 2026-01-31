# 语音助手 Demo

实现「语音输入 → 大模型理解与回复 → 语音输出」的闭环；当前为**文本模式**（终端输入/输出）。

方案见 [语音助手方案.md](./语音助手方案.md)。

## 目录结构

```
voice_assistant/           # 本子项目目录
├── README.md
├── 语音助手方案.md
├── run_voice_assistant.py   # 入口
└── voice_assistant/         # Python 实现包
    ├── __init__.py
    ├── llm_client.py
    ├── pipeline.py
    ├── asr.py
    └── tts.py
```

## 运行方式（文本模式）

在**项目根目录**执行：

```bash
uv run python demo/voice_assistant/run_voice_assistant.py
```

或进入本目录后（需保证项目根在 `PYTHONPATH` 或由脚本添加）：

```bash
uv run python run_voice_assistant.py
```

按提示输入一句话回车，助手会调用 LLM 并打印回复；输入空行退出。

## 环境变量

与主项目一致（如从 `~/.zshrc` 读取）：

- `USE_OLLAMA=true`：使用本地 Ollama
- `OLLAMA_BASE_URL`：默认 `http://localhost:11434`
- `SILICONFLOW_API_KEY`：硅基流动 API（可选）
- `USE_MODELSCOPE`：ModelScope 本地管道（可选）

## 后续

- 阶段二：在 `voice_assistant/asr.py`、`tts.py` 中接入 ModelScope FunASR / TTS，实现完整语音闭环。
