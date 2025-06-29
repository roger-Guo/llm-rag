# RAG系统环境配置

## 环境信息
- **环境名称**: rag_system
- **Python版本**: 3.10.17
- **导出时间**: 2025年6月30日

## 环境文件
- `rag_system_environment.yml` - 完整的Conda环境配置
- `requirements.txt` - Python包依赖列表

## 重现环境

### 方法1: 使用Conda环境文件（推荐）
```bash
# 创建环境
conda env create -f rag_system_environment.yml

# 激活环境
conda activate rag_system
```

### 方法2: 使用requirements.txt
```bash
# 创建新的conda环境
conda create -n rag_system python=3.10

# 激活环境
conda activate rag_system

# 安装依赖
pip install -r requirements.txt
```

## 核心依赖包
- **sentence-transformers==4.1.0** - 文本嵌入模型
- **chromadb==1.0.12** - 向量数据库
- **openai==1.82.1** - OpenAI API客户端
- **scikit-learn==1.6.1** - 机器学习库（TF-IDF）
- **torch==2.2.2** - PyTorch深度学习框架
- **transformers==4.52.4** - Hugging Face转换器库
- **numpy==1.26.4** - 数值计算库

## 环境验证
```bash
# 激活环境
conda activate rag_system

# 验证核心包
python -c "import sentence_transformers; print('✅ sentence-transformers')"
python -c "import chromadb; print('✅ chromadb')"
python -c "import openai; print('✅ openai')"
python -c "import sklearn; print('✅ scikit-learn')"
python -c "import torch; print('✅ torch')"
```

## 项目运行
```bash
# 设置环境变量
export DEEPSEEK_API_KEY="your-api-key"

# 运行测试
python test_qa.py
python debug_data_loading.py
```

## 注意事项
1. 确保设置了正确的API密钥环境变量
2. 首次运行可能需要下载模型文件
3. 建议使用conda环境文件以确保版本一致性 