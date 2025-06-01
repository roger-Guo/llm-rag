#!/bin/bash

# RAG系统启动脚本

echo "🚀 启动RAG智能问答系统..."

# 设置ModelScope镜像源
export MODELSCOPE_CACHE=./models

# 激活conda环境
echo "📦 激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate rag_system

if [ $? -ne 0 ]; then
    echo "❌ 激活conda环境失败，请确保已创建rag_system环境"
    exit 1
fi

# 检查Python环境
echo "🐍 检查Python环境..."
python --version

# 检查必要的包
echo "📋 检查依赖包..."
python -c "import streamlit, chromadb, sentence_transformers, openai, modelscope; print('✅ 所有依赖包已安装')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要的依赖包，请运行: pip install streamlit chromadb sentence-transformers openai modelscope"
    exit 1
fi

# 进入src目录
cd src

# 启动Streamlit应用
echo "📱 正在启动Web界面..."
echo "🌐 访问地址: http://localhost:8501"
echo "💡 提示: 首次运行可能需要下载模型，请耐心等待"
echo "🛑 按 Ctrl+C 停止服务"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 