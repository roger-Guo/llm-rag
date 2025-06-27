#!/usr/bin/env python3
"""
数据加载调试脚本
"""
import sys
import os
sys.path.append('./src')

try:
    from config import Config
    from rag_system import RAGSystem
    from utils import load_toutiao_data, clean_text
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def debug_data_loading():
    """调试数据加载过程"""
    print("🔍 开始调试数据加载...")
    
    # 1. 检查配置
    print("\n1️⃣ 检查配置...")
    try:
        config = Config()
        print(f"✅ 配置加载成功")
        print(f"   - 数据文件: {config.TOUTIAO_DATA_FILE}")
        print(f"   - TF-IDF模式: {getattr(config, 'USE_TFIDF_ONLY', False)}")
        print(f"   - ChromaDB目录: {config.CHROMA_PERSIST_DIR}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 2. 检查数据文件
    print("\n2️⃣ 检查数据文件...")
    data_file = "./data/三国演义.txt"
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file) / 1024 / 1024
        print(f"✅ 数据文件存在: {data_file} ({file_size:.1f}MB)")
        
        # 读取前几行检查格式
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(3)]
            print(f"   - 文件前3行预览:")
            for i, line in enumerate(lines, 1):
                print(f"     {i}: {line[:100]}...")
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
    else:
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    # 3. 测试数据加载函数
    print("\n3️⃣ 测试数据加载函数...")
    try:
        raw_data = load_toutiao_data(data_file, max_lines=5)
        print(f"✅ 数据加载函数工作正常，加载了 {len(raw_data)} 条数据")
        if raw_data:
            print(f"   - 第一条数据: {raw_data[0]}")
    except Exception as e:
        print(f"❌ 数据加载函数失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 初始化RAG系统
    print("\n4️⃣ 初始化RAG系统...")
    try:
        rag_system = RAGSystem(config)
        print("✅ RAG系统创建成功")
        
        # 初始化组件
        success = rag_system.initialize()
        if success:
            print("✅ RAG系统初始化成功")
            print(f"   - 使用ModelScope: {rag_system.using_modelscope}")
            print(f"   - 使用TF-IDF: {rag_system.using_tfidf}")
        else:
            print("❌ RAG系统初始化失败")
            return
    except Exception as e:
        print(f"❌ RAG系统初始化异常: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 测试数据加载到向量库
    print("\n5️⃣ 测试数据加载到向量库...")
    try:
        print("开始加载数据（限制10条）...")
        success = rag_system.load_and_index_data(
            data_file, 
            max_documents=10,  # 只测试10条
            force_reload=True
        )
        
        if success:
            print("✅ 数据加载到向量库成功")
            
            # 检查统计信息
            stats = rag_system.get_collection_stats()
            print(f"   - 统计信息: {stats}")
        else:
            print("❌ 数据加载到向量库失败")
            
    except Exception as e:
        print(f"❌ 数据加载异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔍 调试完成")

if __name__ == "__main__":
    debug_data_loading() 