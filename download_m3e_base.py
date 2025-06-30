#!/usr/bin/env python3
"""
使用ModelScope下载M3E-Base嵌入模型
专门为中文优化的嵌入模型，支持离线模式，适合16G内存环境
模型地址: https://modelscope.cn/models/AI-ModelScope/m3e-base
"""

import os
import sys
import time
from pathlib import Path

def download_m3e_model():
    """使用ModelScope下载M3E-Base模型"""
    
    print("🚀 开始下载M3E-Base模型...")
    print("📋 模型信息:")
    print("   - 名称: AI-ModelScope/m3e-base")
    print("   - 地址: https://modelscope.cn/models/AI-ModelScope/m3e-base")
    print("   - 大小: 约400MB")
    print("   - 维度: 768维")
    print("   - 特点: 专门为中文优化，支持多语言，离线友好")
    print("   - 适用: 16G内存环境，RAG系统")
    print()
    
    try:
        # 导入ModelScope
        from modelscope import snapshot_download
        
        # 确保models目录存在
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        print("📥 正在从ModelScope下载M3E-Base模型...")
        print("⏳ 这可能需要几分钟时间，请耐心等待...")
        
        start_time = time.time()
        
        # 使用ModelScope下载模型 - 使用正确的模型ID
        model_dir = snapshot_download(
            model_id="AI-ModelScope/m3e-base",
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
            
            # 列出主要模型文件
            print("\n📁 模型文件列表:")
            important_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'vocab.txt']
            for file_path in Path(model_dir).iterdir():
                if file_path.is_file():
                    file_size = file_path.stat().st_size / 1024 / 1024
                    status = "🔑" if file_path.name in important_files else "📄"
                    print(f"   {status} {file_path.name}: {file_size:.1f}MB")
        
        print("\n🎉 M3E-Base嵌入模型安装完成！")
        print("💡 现在可以在RAG系统中使用这个模型了")
        
        return model_dir
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 尝试手动下载模型文件")
        print("4. 检查ModelScope依赖是否正确安装")
        return None

def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载...")
    
    try:
        # 尝试使用sentence-transformers加载
        try:
            from sentence_transformers import SentenceTransformer
            
            # 尝试从本地路径加载
            model_paths = [
                "AI-ModelScope/m3e-base",  # ModelScope路径
                "models/AI-ModelScope/m3e-base",  # 本地路径
            ]
            
            model = None
            for path in model_paths:
                try:
                    print(f"🔄 尝试加载路径: {path}")
                    model = SentenceTransformer(path)
                    print(f"✅ 成功从路径加载: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ 路径 {path} 加载失败: {e}")
                    continue
            
            if model is None:
                print("❌ 所有路径都加载失败")
                return False
            
            # 测试编码
            test_texts = [
                "这是一个测试句子",
                "今天天气很好",
                "M3E-Base是一个优秀的中文嵌入模型",
                "RAG系统需要高质量的嵌入向量"
            ]
            
            print("🔍 正在测试编码功能...")
            embeddings = model.encode(test_texts)
            print(f"✅ 模型编码测试成功！")
            print(f"   - 维度: {embeddings.shape[1]}")
            print(f"   - 输入文本数量: {len(test_texts)}")
            print(f"   - 嵌入形状: {embeddings.shape}")
            
            # 测试相似度计算
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(embeddings[:2])
            print(f"   - 示例相似度: {similarity[0][1]:.4f}")
            
        except ImportError:
            print("⚠️ sentence-transformers未安装，跳过测试")
        except Exception as e:
            print(f"⚠️ sentence-transformers加载失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def update_config():
    """更新配置文件以使用M3E-Base模型"""
    print("\n⚙️ 更新配置文件...")
    
    try:
        config_path = "src/config.py"
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换模型名称
        old_patterns = [
            'EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"',
            'EMBEDDING_MODEL_NAME = "moka-ai/m3e-base"'
        ]
        
        new_value = 'EMBEDDING_MODEL_NAME = "AI-ModelScope/m3e-base"'
        
        new_content = content
        for pattern in old_patterns:
            if pattern in content:
                new_content = new_content.replace(pattern, new_value)
                break
        else:
            # 如果没找到现有配置，添加注释
            print("⚠️ 未找到现有模型配置，请手动更新")
            return False
        
        # 写回配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 配置文件已更新为使用M3E-Base模型")
        print(f"✅ 新配置: {new_value}")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件更新失败: {e}")
        return False

def test_rag_integration():
    """测试RAG系统集成"""
    print("\n🔍 测试RAG系统集成...")
    
    try:
        # 导入RAG系统
        sys.path.append("src")
        from rag_system import RAGSystem
        from config import Config
        
        # 创建配置
        config = Config()
        print(f"📋 当前配置的模型: {config.EMBEDDING_MODEL_NAME}")
        
        # 测试模型加载
        print("📋 测试RAG系统模型加载...")
        rag = RAGSystem(config)
        
        # 初始化系统
        print("🔧 初始化RAG系统...")
        success = rag.initialize()
        if success:
            print("✅ RAG系统初始化成功！")
            print(f"✅ 使用模型: {config.EMBEDDING_MODEL_NAME}")
            
            # 测试简单的嵌入生成
            if hasattr(rag, 'embedding_model') and rag.embedding_model:
                test_text = "这是一个测试文本"
                embedding = rag.embedding_model.encode([test_text])
                print(f"✅ 嵌入生成测试成功，维度: {embedding.shape[1]}")
            
            return True
        else:
            print("❌ RAG系统初始化失败")
            return False
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def manual_download_guide():
    """提供手动下载指南"""
    print("\n" + "=" * 60)
    print("📋 手动下载指南（如果自动下载失败）:")
    print("=" * 60)
    print("1. 访问: https://modelscope.cn/models/AI-ModelScope/m3e-base")
    print("2. 点击'Files and versions'标签")
    print("3. 下载以下关键文件:")
    print("   - config.json")
    print("   - pytorch_model.bin")
    print("   - sentence_bert_config.json")
    print("   - tokenizer.json")
    print("   - tokenizer_config.json")
    print("   - vocab.txt")
    print("4. 创建目录: models/AI-ModelScope/m3e-base/")
    print("5. 将下载的文件放入该目录")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 M3E-Base 模型下载器 (ModelScope)")
    print("🌐 模型地址: https://modelscope.cn/models/AI-ModelScope/m3e-base")
    print("=" * 60)
    
    # 下载模型
    model_dir = download_m3e_model()
    
    if model_dir:
        # 测试模型加载
        test_success = test_model_loading()
        
        if test_success:
            # 更新配置文件
            config_success = update_config()
            
            if config_success:
                # 测试RAG集成
                test_rag_integration()
    else:
        # 提供手动下载指南
        manual_download_guide()
    
    print("\n" + "=" * 60)
    print("📝 M3E-Base模型特点:")
    print("1. 🇨🇳 专门为中文优化")
    print("2. 🔌 支持离线模式")
    print("3. 💾 适合16G内存环境")
    print("4. 🎯 768维高精度嵌入")
    print("5. 🚀 在RAG系统中表现优异")
    print("6. 📚 支持多种文本任务")
    print("\n📁 模型文件位置: models/AI-ModelScope/m3e-base/")
    print("⚙️ 配置文件: src/config.py")
    print("=" * 60) 