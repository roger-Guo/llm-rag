#!/usr/bin/env python3
"""
使用ModelScope下载BGE-small-zh-v1.5嵌入模型
简化版本，直接下载模型
"""

import os
import sys
import time
from pathlib import Path

def download_bge_model():
    """使用ModelScope下载BGE-small-zh-v1.5模型"""
    
    print("🚀 开始下载BGE-small-zh-v1.5模型...")
    print("📋 模型信息:")
    print("   - 名称: BAAI/bge-small-zh-v1.5")
    print("   - 大小: 约100MB")
    print("   - 维度: 384维")
    print("   - 特点: 专门为中文优化，内存友好")
    print()
    
    try:
        # 导入ModelScope
        from modelscope import snapshot_download
        
        # 确保models目录存在
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        print("📥 正在从ModelScope下载BGE-small-zh-v1.5模型...")
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
        
        # 检查模型文件
        if Path(model_dir).exists():
            total_size = sum(f.stat().st_size for f in Path(model_dir).rglob('*') if f.is_file())
            size_mb = total_size / 1024 / 1024
            print(f"📊 模型文件大小: {size_mb:.1f}MB")
            
            # 列出模型文件
            print("\n📁 模型文件列表:")
            for file_path in Path(model_dir).iterdir():
                if file_path.is_file():
                    file_size = file_path.stat().st_size / 1024 / 1024
                    print(f"   - {file_path.name}: {file_size:.1f}MB")
        
        print("\n🎉 中文嵌入模型安装完成！")
        print("💡 现在可以在RAG系统中使用这个模型了")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 尝试手动下载模型文件")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载...")
    
    try:
        # 尝试使用sentence-transformers加载
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            print("✅ 使用sentence-transformers成功加载模型")
            
            # 测试编码
            test_texts = ["这是一个测试句子", "今天天气很好"]
            embeddings = model.encode(test_texts)
            print(f"✅ 模型编码测试成功，维度: {embeddings.shape[1]}")
            
        except ImportError:
            print("⚠️ sentence-transformers未安装，跳过测试")
        except Exception as e:
            print(f"⚠️ sentence-transformers加载失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 BGE-small-zh-v1.5 模型下载器 (ModelScope简化版)")
    print("=" * 60)
    
    # 下载模型
    success = download_bge_model()
    
    if success:
        # 测试模型加载
        test_model_loading()
    
    print("\n" + "=" * 60)
    print("📝 使用说明:")
    print("1. 模型已下载到 models/ 目录")
    print("2. RAG系统会自动使用这个模型")
    print("3. 模型路径已配置为: BAAI/bge-small-zh-v1.5")
    print("=" * 60) 