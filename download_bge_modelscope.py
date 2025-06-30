#!/usr/bin/env python3
"""
使用ModelScope下载BGE-small-zh-v1.5嵌入模型
专门为中文优化的轻量级嵌入模型，适合16G内存环境
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def install_modelscope():
    """安装ModelScope依赖"""
    print("📦 安装ModelScope依赖...")
    
    packages = [
        "modelscope",
        "sentence-transformers",
        "transformers",
        "torch"
    ]
    
    for package in packages:
        try:
            print(f"📥 安装 {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
            return False
    
    return True

def download_bge_model():
    """使用ModelScope下载BGE-small-zh-v1.5模型"""
    
    print("\n🚀 开始下载BGE-small-zh-v1.5模型...")
    print("📋 模型信息:")
    print("   - 名称: BAAI/bge-small-zh-v1.5")
    print("   - 大小: 约100MB")
    print("   - 维度: 384维")
    print("   - 特点: 专门为中文优化，内存友好")
    print()
    
    try:
        # 确保models目录存在
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # 导入必要的库
        print("📦 检查依赖...")
        try:
            from modelscope import snapshot_download
            from sentence_transformers import SentenceTransformer
            print("✅ ModelScope 和 sentence-transformers 已安装")
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            return False
        
        # 下载模型
        print("\n📥 正在从ModelScope下载BGE-small-zh-v1.5模型...")
        print("⏳ 这可能需要几分钟时间，请耐心等待...")
        
        start_time = time.time()
        
        # 使用ModelScope下载模型
        model_dir = snapshot_download(
            model_id="BAAI/bge-small-zh-v1.5",
            cache_dir="models",
            revision="master"
        )
        
        end_time = time.time()
        download_time = end_time - start_time
        
        print(f"✅ 模型下载完成！")
        print(f"📁 模型路径: {model_dir}")
        print(f"⏱️ 耗时: {download_time:.1f}秒")
        
        # 测试模型
        print("\n🧪 测试模型功能...")
        try:
            # 使用下载的模型路径
            model = SentenceTransformer(model_dir)
            
            test_texts = [
                "这是一个测试句子",
                "这是另一个测试句子",
                "今天天气很好"
            ]
            
            embeddings = model.encode(test_texts)
            print(f"✅ 模型测试成功！")
            print(f"   - 输入文本数量: {len(test_texts)}")
            print(f"   - 嵌入维度: {embeddings.shape[1]}")
            print(f"   - 嵌入形状: {embeddings.shape}")
            
        except Exception as e:
            print(f"⚠️ 模型测试失败: {e}")
            print("尝试使用HuggingFace Hub下载...")
            
            # 如果ModelScope下载的模型有问题，尝试直接使用HuggingFace
            try:
                model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
                print("✅ 使用HuggingFace Hub成功加载模型")
            except Exception as e2:
                print(f"❌ HuggingFace Hub也失败: {e2}")
                return False
        
        # 检查模型文件
        if Path(model_dir).exists():
            total_size = sum(f.stat().st_size for f in Path(model_dir).rglob('*') if f.is_file())
            size_mb = total_size / 1024 / 1024
            print(f"📊 模型文件大小: {size_mb:.1f}MB")
        
        print("\n🎉 中文嵌入模型安装完成！")
        print("💡 现在可以在RAG系统中使用这个模型了")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 尝试手动下载模型文件")
        print("4. 检查Python环境")
        return False

def test_model_integration():
    """测试模型在RAG系统中的集成"""
    print("\n🔍 测试模型集成...")
    
    try:
        # 导入RAG系统
        sys.path.append("src")
        from rag_system import RAGSystem
        from config import Config
        
        # 创建配置
        config = Config()
        
        # 测试模型加载
        print("📋 测试RAG系统模型加载...")
        rag = RAGSystem(config)
        
        # 测试嵌入功能
        test_query = "诸葛亮是谁？"
        print(f"🔍 测试查询: {test_query}")
        
        # 获取嵌入
        query_embedding = rag.get_query_embedding(test_query)
        print(f"✅ 查询嵌入成功，维度: {len(query_embedding)}")
        
        print("🎉 模型集成测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def manual_download_guide():
    """提供手动下载指南"""
    print("\n" + "=" * 60)
    print("📋 手动下载指南（如果自动下载失败）:")
    print("=" * 60)
    print("1. 访问: https://modelscope.cn/models/BAAI/bge-small-zh-v1.5/summary")
    print("2. 点击'下载模型'按钮")
    print("3. 下载完成后解压到以下目录:")
    print("   ~/.cache/torch/sentence_transformers/BAAI_bge-small-zh-v1.5/")
    print("4. 确保包含以下文件:")
    print("   - config.json")
    print("   - pytorch_model.bin")
    print("   - sentence_bert_config.json")
    print("   - tokenizer.json")
    print("   - tokenizer_config.json")
    print("   - vocab.txt")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 BGE-small-zh-v1.5 模型下载器 (ModelScope)")
    print("=" * 60)
    
    # 安装依赖
    if not install_modelscope():
        print("❌ 依赖安装失败，退出")
        sys.exit(1)
    
    # 下载模型
    success = download_bge_model()
    
    if success:
        # 测试集成
        test_model_integration()
    else:
        # 提供手动下载指南
        manual_download_guide()
    
    print("\n" + "=" * 60)
    print("📝 使用说明:")
    print("1. 模型已下载到本地缓存")
    print("2. RAG系统会自动使用这个模型")
    print("3. 可以通过Web界面切换模型")
    print("4. 模型文件位置: models/ 或 ~/.cache/torch/sentence_transformers/")
    print("=" * 60) 