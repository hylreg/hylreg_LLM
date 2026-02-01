#!/usr/bin/env python3
"""
视觉语言 (VL) Demo 入口脚本
用法: uv run python demo/vl_demo/run_vl_demo.py --image <图片路径> [--question "问题"]
从项目根目录运行: uv run python demo/vl_demo/run_vl_demo.py
"""

import argparse
import sys
from pathlib import Path

_file = Path(__file__).resolve()
_demo_dir = _file.parents[1]  # demo/，使 vl_demo 解析为 demo/vl_demo

if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

from vl_demo.vl import VLClient


def main() -> None:
    parser = argparse.ArgumentParser(description="VL Demo: 图片 + 问题 → VL 模型回复")
    parser.add_argument("--image", "-i", required=True, help="本地图片路径")
    parser.add_argument(
        "--question", "-q",
        default="请描述这张图片",
        help="对图片的提问（默认：请描述这张图片）",
    )
    parser.add_argument(
        "--model", "-m",
        default="",
        help="VL 模型名（可选；不填则用后端默认）",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"错误：图片不存在: {image_path}", file=sys.stderr)
        sys.exit(1)

    client = VLClient(model=args.model or None)
    try:
        answer = client.ask(str(image_path), question=args.question)
        print(answer or "(无回复)")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
