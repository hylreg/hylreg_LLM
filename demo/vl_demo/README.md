# 视觉语言 (VL) Demo

本地图片 + 文本问题 → VL 模型理解 → 文本回复。仅使用 **ollama**、**siliconflow**、**modelscope** 的 API，优先 **Qwen 系列 VL 模型**。

## 运行方式

在**项目根目录**执行：

```bash
uv run python demo/vl_demo/run_vl_demo.py --image /path/to/image.jpg
```

带自定义问题：

```bash
uv run python demo/vl_demo/run_vl_demo.py --image /path/to/image.jpg --question "图中有什么文字？"
```

## 环境变量

- **Ollama**：`USE_OLLAMA=true`，并确保本机已启动 Ollama 且已拉取视觉模型（如 `ollama pull llava` 或 `ollama pull qwen2-vl`）。
- **SiliconFlow**：在 `~/.zshrc` 中设置 `SILICONFLOW_API_KEY`，可选 `SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/`。Demo 会使用云端 VL 模型（如 Qwen2.5-VL）。

未设置 SiliconFlow 且未设置 `USE_OLLAMA=true` 时会报错并提示配置方式。

## 目录结构

```
vl_demo/
├── README.md              # 本说明
├── 视觉语言Demo方案.md     # 技术方案
├── run_vl_demo.py         # 入口脚本
└── vl_demo/
    ├── __init__.py
    └── vl_client.py       # VL 客户端
```

## 技术方案

详见 [视觉语言Demo方案.md](./视觉语言Demo方案.md)。
