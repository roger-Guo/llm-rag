# RAG项目调试指南

## 🚀 快速开始调试

### 1. 环境检查
```bash
# 检查Python环境
python --version
pip list | grep -E "(sentence-transformers|chromadb|openai|sklearn)"

# 检查环境变量
echo $DEEPSEEK_API_KEY
```

### 2. 基础调试脚本

#### 数据加载调试
```bash
python debug_data_loading.py
```
这个脚本会：
- ✅ 检查配置加载
- ✅ 验证数据文件存在
- ✅ 测试数据加载函数
- ✅ 初始化RAG系统
- ✅ 测试向量库索引

#### 问答功能测试
```bash
python test_qa.py
```
这个脚本会：
- ✅ 初始化完整系统
- ✅ 加载测试数据
- ✅ 执行多个测试问题
- ✅ 显示性能指标

#### 搜索功能测试
```bash
python test_search.py
```
这个脚本会：
- ✅ 测试搜索功能
- ✅ 验证TF-IDF和嵌入向量搜索

## 🔧 高级调试方法

### 1. 使用Python调试器 (pdb)

#### 在代码中添加断点
```python
import pdb; pdb.set_trace()  # 在需要调试的地方添加这行
```

#### 启动调试模式
```bash
python -m pdb your_script.py
```

#### 常用pdb命令
- `n` (next): 执行下一行
- `s` (step): 步入函数
- `c` (continue): 继续执行
- `l` (list): 显示当前代码
- `p variable`: 打印变量值
- `q` (quit): 退出调试器

### 2. 使用VS Code调试

#### 创建launch.json配置
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug RAG System",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/test_qa.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "DEEPSEEK_API_KEY": "your-api-key"
            }
        }
    ]
}
```

### 3. 日志调试

#### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 在特定模块中启用调试
```python
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

## 🐛 常见问题调试

### 1. 模块导入错误
```bash
# 检查Python路径
python -c "import sys; print(sys.path)"

# 添加src目录到路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 2. 环境变量问题
```bash
# 检查环境变量
env | grep DEEPSEEK

# 临时设置环境变量
export DEEPSEEK_API_KEY="your-api-key"
```

### 3. 模型下载问题
```bash
# 检查模型缓存目录
ls -la models/

# 清理缓存重新下载
rm -rf models/*
```

### 4. ChromaDB问题
```bash
# 检查ChromaDB数据
ls -la chroma_db/

# 重置ChromaDB
rm -rf chroma_db/*
```

## 📊 性能调试

### 1. 内存使用监控
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### 2. 执行时间分析
```python
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.2f}s")
        return result
    return wrapper
```

### 3. 模型大小检查
```python
def check_model_size(model_path):
    import os
    size = os.path.getsize(model_path) / 1024 / 1024
    print(f"模型大小: {size:.1f} MB")
```

## 🔍 交互式调试

### 1. 使用IPython/Jupyter
```python
# 在代码中添加
import IPython; IPython.embed()
```

### 2. 创建调试脚本
```python
#!/usr/bin/env python3
"""
交互式调试脚本
"""
import sys
sys.path.append('./src')

from config import Config
from rag_system import RAGSystem

# 初始化系统
config = Config()
rag_system = RAGSystem(config)
rag_system.initialize()

# 进入交互模式
import IPython
IPython.embed()
```

## 📝 调试检查清单

### 系统初始化
- [ ] Python环境正确
- [ ] 所有依赖已安装
- [ ] 环境变量已设置
- [ ] 数据文件存在
- [ ] 模型缓存目录可写

### 功能测试
- [ ] 配置加载正常
- [ ] 数据加载成功
- [ ] 向量库初始化
- [ ] 搜索功能正常
- [ ] 问答功能正常

### 性能检查
- [ ] 内存使用合理
- [ ] 响应时间可接受
- [ ] 模型加载速度
- [ ] 搜索结果质量

## 🆘 获取帮助

### 1. 查看错误日志
```bash
# 运行脚本并保存日志
python your_script.py 2>&1 | tee debug.log
```

### 2. 检查系统信息
```bash
# 系统信息
python -c "import platform; print(platform.platform())"

# Python包信息
pip freeze > requirements.txt
```

### 3. 最小化复现
创建一个最小化的测试脚本来复现问题：
```python
#!/usr/bin/env python3
"""
最小化测试脚本
"""
import sys
sys.path.append('./src')

# 只导入必要的模块
from config import Config

# 测试最小功能
config = Config()
print("配置加载成功")
``` 