# 🤖 RAG智能问答系统

基于西游记小说的智能问答系统，使用RAG（检索增强生成）技术，集成ModelScope镜像源、ChromaDB向量数据库和DeepSeek API。

## ✨ 功能特点

- 🚀 **ModelScope镜像源**: 支持中文嵌入模型m3e-base
- 🧠 **智能问答**: 基于DeepSeek API的智能回答生成
- 🔍 **语义搜索**: 支持向量搜索
- 📊 **实时监控**: 性能指标和系统状态监控
- 🎯 **信息溯源**: 准确的信息来源追踪
- 🌐 **Web界面**: 现代化的Streamlit用户界面

## 🛠️ 技术架构

- **前端**: Streamlit Web界面
- **嵌入模型**: ModelScope中文嵌入模型 (AI-ModelScope/m3e-base)
- **向量数据库**: ChromaDB
- **语言模型**: DeepSeek Chat API

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Conda环境: rag_system

### 2. 启动系统

```bash

# 方法一：使用启动脚本（会尝试下载嵌入模型）
./start_rag.sh

# 方法二：手动启动
cd src
conda activate rag_system
export MODELSCOPE_CACHE=../models
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 3. 访问界面

打开浏览器访问: http://localhost:8501

## 🎯 嵌入模型选择

### ⚡ TF-IDF模式（推荐新手）
- ✅ **无需下载**: 不需要下载任何模型文件
- ✅ **启动最快**: 几秒钟即可启动
- ✅ **资源占用小**: 内存和磁盘占用最少  
- ✅ **稳定可靠**: 传统算法，兼容性好
- ⚠️ **效果**: 相比深度学习模型效果略差，但对中文支持良好

## 📖 使用指南

### 初次使用

1. **初始化系统**: 在侧边栏点击"🚀 初始化系统"
   - 首次运行会下载中文嵌入模型（约400MB）

2. **生成示例数据**: 点击"生成示例数据"创建测试数据

3. **加载数据**: 点击"加载数据到向量库"建立索引

### 智能问答

1. 在问答界面输入问题
2. 系统会检索相关信息并生成智能回答
3. 可查看参考来源和性能指标

## ⚙️ 配置说明

### DeepSeek API配置

在 `src/config.py` 中配置：

```python
DEEPSEEK_API_KEY = "sk-your-api-key"  # API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
```

### 模型配置

```python
MODEL_CACHE_DIR = "../models"  # 模型缓存目录
EMBEDDING_MODEL_NAME = "AI-ModelScope/m3e-base"  # 备选模型
```

### 数据配置

```python
MAX_CHUNK_SIZE = 500      # 文本块最大长度
CHUNK_OVERLAP = 50        # 文本块重叠长度
DEFAULT_TOP_K = 5         # 默认检索结果数量
```

## 🔧 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 可手动清理models目录重试

2. **向量数据库错误**
   - 删除chroma_db目录重新初始化
   - 检查磁盘空间

3. **API调用失败**
   - 检查DeepSeek API密钥
   - 验证网络连接
   - 系统会提供基础检索结果

### 性能优化

- 首次加载建议限制文档数量（100-1000条）
- 可调整检索结果数量(top_k)平衡性能和准确性
- 大数据集建议分批加载

## 📁 项目结构

```
rag_project/
├── src/
│   ├── app.py           # Streamlit Web应用
│   ├── rag_system.py    # RAG系统核心
│   ├── config.py        # 系统配置
│   └── utils.py         # 工具函数
├── data/                # 数据目录
├── models/              # 模型缓存
├── chroma_db/           # 向量数据库
├── start_rag.sh         # 启动脚本
└── README.md            # 说明文档
```

## 🎯 系统状态

- ✅ 系统已初始化
- ✅ Web界面运行在 http://localhost:8501
- ✅ ModelScope镜像源配置完成
- ✅ DeepSeek API已配置

## 📝 更新日志

### v1.0.0 (2025-06-20)
- ✅ 基础RAG系统实现
- ✅ ModelScope镜像源集成
- ✅ Streamlit Web界面
- ✅ DeepSeek API集成
- ✅ 完整错误处理和日志系统

---

🚀 **享受智能问答的便利！** 如有问题，请查看日志或重新初始化系统。 