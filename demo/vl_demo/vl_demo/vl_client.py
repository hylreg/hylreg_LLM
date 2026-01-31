"""
VL 客户端：本地图片 + 文本问题 → 调用 Ollama / SiliconFlow 视觉 API → 返回文本回复
仅使用 ollama、siliconflow，优先 Qwen 系列 VL 模型。
"""

import base64
import os
from pathlib import Path
from typing import Optional

# 默认模型名（Ollama / SiliconFlow 各自默认）
DEFAULT_OLLAMA_VL_MODEL = "llava"
DEFAULT_SILICONFLOW_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def _image_to_base64(image_path: str) -> str:
    """读取本地图片并转为 base64（无 data URL 前缀）。"""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("utf-8")


def _mime_from_suffix(path: Path) -> str:
    """根据后缀返回常见 MIME。"""
    suffix = path.suffix.lower()
    if suffix in (".png",):
        return "image/png"
    if suffix in (".gif",):
        return "image/gif"
    if suffix in (".webp",):
        return "image/webp"
    return "image/jpeg"


class VLClient:
    """
    视觉语言客户端：根据环境变量选择 Ollama 或 SiliconFlow，发送图片+问题，返回模型回复文本。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self._backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        """解析使用哪个后端：ollama 或 siliconflow。"""
        use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"
        siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        if use_ollama:
            return "ollama"
        if siliconflow_api_key:
            return "siliconflow"
        raise ValueError(
            "未找到可用的 VL 配置。请设置 USE_OLLAMA=true 或提供 SILICONFLOW_API_KEY（可从 ~/.zshrc 环境变量读取）。"
        )

    def _get_model(self) -> str:
        if self.model:
            return self.model
        if self._backend == "ollama":
            return DEFAULT_OLLAMA_VL_MODEL
        return DEFAULT_SILICONFLOW_VL_MODEL

    def ask(self, image_path: str, question: str = "请描述这张图片") -> str:
        """
        发送本地图片与文本问题到 VL 模型，返回回复文本。
        """
        b64 = _image_to_base64(image_path)
        path = Path(image_path)
        mime = _mime_from_suffix(path)
        model_name = self._get_model()

        if self._backend == "ollama":
            return self._ask_ollama(b64, question, model_name)
        return self._ask_siliconflow(b64, mime, question, model_name)

    def _ask_ollama(self, b64: str, question: str, model: str) -> str:
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [b64],
                }
            ],
            "stream": False,
        }
        if self.verbose:
            print(f"使用 Ollama: {base_url}, 模型: {model}")
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()

    def _ask_siliconflow(self, b64: str, mime: str, question: str, model: str) -> str:
        """SiliconFlow：OpenAI 兼容 chat completions，content 为 image_url + text。"""
        try:
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError("请安装 langchain-openai、langchain-core: uv add langchain-openai langchain-core") from e

        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/").rstrip("/")
        if self.verbose:
            print(f"使用 SiliconFlow: {base_url}, 模型: {model}")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )
        # OpenAI 多模态格式：content 为 list，先图后文
        url = f"data:{mime};base64,{b64}"
        content = [
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
            {"type": "text", "text": question},
        ]
        msg = HumanMessage(content=content)
        response = llm.invoke([msg])
        return (response.content or "").strip()
