#!/usr/bin/env python3
"""
修复示例文件中的导入路径
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """修复单个文件中的导入路径"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复文档字符串中的错误插入
    content = re.sub(
        r'"""\nimport sys\nfrom pathlib import Path\nsys\.path\.insert\(0, str\(Path\(__file__\)\.parent\.parent\.parent\)\)\n',
        '"""\n',
        content
    )
    
    # 将旧导入改为 demo 包路径
    content = content.replace('from RAG.rag_system import', 'from demo.RAG.rag_system import')
    content = content.replace('from agents.agent_system import', 'from demo.agents.agent_system import')
    # 确保有项目根目录到路径（供 demo 包导入）
    if ('from demo.RAG.rag_system import' in content or 'from demo.agents.agent_system import' in content) and 'sys.path.insert(0' not in content:
        import_match = re.search(r'(import os|from pathlib)', content)
        if import_match:
            insert_pos = import_match.start()
            path_setup = 'import sys\nfrom pathlib import Path\n\n# 添加项目根目录到路径\nsys.path.insert(0, str(Path(__file__).parent.parent.parent))\n\n'
            content = content[:insert_pos] + path_setup + content[insert_pos:]
    
    # 修复路径引用（documents 和 vectorstore）
    project_root_pattern = r'project_root = Path\(__file__\)\.parent\.parent\.parent'
    if 'project_root =' not in content:
        # 查找 script_dir 或类似的定义
        script_dir_pattern = r'script_dir = Path\(__file__\)\.parent'
        if re.search(script_dir_pattern, content):
            # 替换为 project_root
            content = re.sub(
                script_dir_pattern,
                'project_root = Path(__file__).parent.parent.parent',
                content
            )
            # 更新 documents_path
            content = re.sub(
                r'documents_path = (script_dir|project_root) / "documents"',
                'documents_path = project_root / "demo" / "RAG" / "documents"',
                content
            )
            # 更新 vectorstore_path
            content = re.sub(
                r'vectorstore_path = (script_dir|project_root) / "vectorstore"',
                'vectorstore_path = project_root / "demo" / "RAG" / "vectorstore"',
                content
            )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 已修复: {file_path}")
        return True
    return False

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"
    
    fixed_count = 0
    for py_file in examples_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\n完成！共修复 {fixed_count} 个文件。")

if __name__ == "__main__":
    main()
