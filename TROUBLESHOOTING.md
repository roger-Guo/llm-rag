# 🔧 RAG系统故障排除指南

## 🌐 Web界面控制台错误

### ❌ "Unchecked runtime.lastError: The message port closed before a response was received."

**问题分析：**
这个错误通常**不是RAG应用的问题**，而是浏览器相关问题：

1. **浏览器扩展冲突**: Chrome扩展（如广告拦截器、翻译工具等）与Streamlit冲突
2. **浏览器缓存问题**: 旧的缓存数据干扰
3. **端口冲突**: 8501端口被其他服务占用

**解决方案：**

#### 🔧 方案1：清理浏览器
```bash
# 清理浏览器缓存和Cookies
# 或者使用无痕/隐私浏览模式访问 http://localhost:8501
```

#### 🔧 方案2：禁用浏览器扩展
- 临时关闭所有Chrome扩展
- 或者使用其他浏览器（Firefox、Safari等）

#### 🔧 方案3：检查端口占用
```bash
# 检查8501端口是否被占用
lsof -i :8501

# 如果被占用，终止进程或使用其他端口
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

## 🐍 Python依赖问题

### ❌ NumPy版本兼容性错误

**错误信息：**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**解决方案：**
```bash
# 降级NumPy到兼容版本
pip install "numpy<2"
```

### ❌ 缺少依赖包

**检查依赖：**
```bash
python -c "import streamlit, chromadb, sentence_transformers, openai, modelscope; print('✅ 所有依赖包已安装')"
```

**安装缺失的包：**
```bash
pip install streamlit chromadb sentence-transformers openai modelscope scikit-learn
```

## 🚀 启动脚本问题

### ❌ 脚本执行权限错误

**解决方案：**
```bash
# 给脚本添加执行权限
chmod +x start_rag.sh

# 检查脚本第一行是否正确
head -1 start_rag.sh
# 应该显示: #!/bin/bash
```

### ❌ Conda环境问题

**解决方案：**
```bash
# 手动激活环境
conda activate rag_system

# 检查Python版本
python --version

# 如果环境不存在，重新创建
conda create -n rag_system python=3.10 -y
```

## 🤖 RAG系统问题

### ❌ ModelScope下载失败

**症状：** 模型下载失败或网络超时

**解决方案：**
1. **检查网络连接**
2. **系统会自动降级到TF-IDF方案**
3. **手动清理models目录重试**：
```bash
rm -rf models/*
```

### ❌ ChromaDB错误

**症状：** 向量数据库初始化失败

**解决方案：**
```bash
# 删除ChromaDB目录重新初始化
rm -rf chroma_db/
```

### ❌ DeepSeek API错误

**症状：** API调用失败

**解决方案：**
1. **检查API密钥**：确保 `src/config.py` 中的密钥正确
2. **检查网络连接**
3. **系统会提供基础检索结果**（即使API失败）

## 🌐 网络和代理问题

### ❌ 连接超时

**解决方案：**
```bash
# 设置代理（如果需要）
export http_proxy=your_proxy_url
export https_proxy=your_proxy_url

# 或者使用镜像源
export MODELSCOPE_CACHE=./models
```

## 📊 性能优化

### ⚠️ 内存不足

**解决方案：**
1. **减少文档数量**：首次加载建议100-1000条
2. **调整批处理大小**：在 `rag_system.py` 中修改 `batch_size`
3. **使用TF-IDF方案**：如果内存严重不足

### ⚠️ 响应速度慢

**解决方案：**
1. **减少检索结果数量**：调整 `top_k` 参数
2. **优化分块大小**：在 `config.py` 中调整 `MAX_CHUNK_SIZE`
3. **使用本地模型**：避免网络延迟

## 🔍 调试方法

### 📋 系统状态检查

```bash
# 检查Streamlit进程
ps aux | grep streamlit

# 检查端口占用
lsof -i :8501

# 检查Python环境
which python
python --version

# 检查当前目录
pwd
ls -la
```

### 📝 日志查看

```bash
# 查看Streamlit日志
# 日志会在终端显示，注意查看错误信息

# 手动启动查看详细日志
cd rag_project/src
conda activate rag_system
export MODELSCOPE_CACHE=../models
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --logger.level debug
```

## 🚀 完整重启流程

如果以上方法都无效，可以尝试完整重启：

```bash
# 1. 终止所有相关进程
pkill -f streamlit

# 2. 清理缓存
rm -rf models/*
rm -rf chroma_db/*

# 3. 重新安装依赖
pip install --upgrade streamlit chromadb sentence-transformers openai modelscope
pip install "numpy<2"

# 4. 重新启动
cd rag_project
./start_rag.sh
```

## 📞 获取帮助

如果问题仍然存在：

1. **检查系统日志**：查看终端输出的详细错误信息
2. **使用无痕浏览器**：排除浏览器扩展问题
3. **尝试不同端口**：避免端口冲突
4. **检查网络连接**：确保可以访问外部服务

---

💡 **提示：** 大多数浏览器控制台错误（如 "message port closed"）不会影响RAG系统的实际功能，可以忽略这些错误继续使用系统。 