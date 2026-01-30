#!/usr/bin/env python3
"""
从 ModelScope 下载 Qwen/Qwen3-Reranker-0.6B 模型
"""

from modelscope import snapshot_download
import os

def download_qwen_reranker(cache_dir='./models'):
    """
    下载 Qwen/Qwen3-Reranker-0.6B 模型
    
    Args:
        cache_dir: 模型缓存目录
    """
    model_name = 'Qwen/Qwen3-Reranker-0.6B'
    
    print(f"开始下载模型: {model_name}")
    print(f"缓存目录: {cache_dir}")
    print("-" * 60)
    
    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        model_dir = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            revision='master'
        )
        print(f"\n✓ 模型下载成功！")
        print(f"  模型路径: {model_dir}")
        
        # 计算模型大小
        def get_dir_size(path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
            return total
        
        size_bytes = get_dir_size(model_dir)
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)
        
        if size_gb >= 1:
            print(f"  模型大小: {size_gb:.2f} GB")
        else:
            print(f"  模型大小: {size_mb:.2f} MB")
        
        return model_dir
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n提示:")
        print("1. 请检查网络连接")
        print("2. 确认模型名称是否正确")
        print("3. 检查磁盘空间是否充足")
        print("4. 如果模型需要认证，请设置 MODELSCOPE_API_KEY 环境变量")
        return None

if __name__ == '__main__':
    print("=" * 60)
    print("Qwen/Qwen3-Reranker-0.6B 模型下载工具")
    print("=" * 60)
    print()
    
    # 可以自定义缓存目录
    cache_dir = os.getenv('MODEL_CACHE_DIR', './models')
    
    model_dir = download_qwen_reranker(cache_dir=cache_dir)
    
    if model_dir:
        print("\n" + "=" * 60)
        print("下载完成！您现在可以使用以下路径加载模型：")
        print(f"  {model_dir}")
        print("=" * 60)