# Demo

本目录存放多个独立 Demo 项目，每个子项目一个文件夹，自带入口与说明。

## 结构约定

```
demo/
├── README.md              # 本总览
├── voice_assistant/       # 子项目：语音助手
│   ├── README.md          # 项目说明与运行方式
│   ├── run_voice_assistant.py
│   └── voice_assistant/   # 实现包
│       └── ...
├── vl_demo/               # 子项目：视觉语言 Demo
│   ├── README.md
│   ├── 视觉语言Demo方案.md
│   ├── run_vl_demo.py
│   └── vl_demo/           # 实现包
│       └── ...
└── <其他子项目>/          # 后续可继续添加
    ├── README.md
    ├── run_*.py
    └── ...
```

- 每个子项目在 `demo/` 下占一个目录，目录名即项目名。
- 子项目内包含：`README.md`、入口脚本（如 `run_xxx.py`）、实现代码（可再包一层同名或 `src` 包）。
- 运行方式：在**项目根目录**执行 `uv run python demo/<子项目>/run_xxx.py`，或进入子项目目录后按该子项目 README 执行。

## 当前子项目

| 项目 | 说明 | 运行 |
|------|------|------|
| [voice_assistant](./voice_assistant/) | 语音助手（ASR → LLM → TTS，当前为文本模式） | `uv run python demo/voice_assistant/run_voice_assistant.py` |
| [vl_demo](./vl_demo/) | 视觉语言 Demo（图片 + 问题 → VL 模型 → 文本回复） | `uv run python demo/vl_demo/run_vl_demo.py --image <图片路径>` |

## 添加新项目

在 `demo/` 下新建目录，例如 `my_demo/`，放入：

- `README.md`：项目说明与运行方式
- 入口脚本与实现代码

并在本 README 的「当前子项目」表中增加一行即可。
