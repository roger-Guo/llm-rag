#!/usr/bin/env python3
"""
搜索功能测试脚本
"""
import sys
import os
sys.path.append('./src')

from config import Config
from rag_system import RAGSystem

def test_search():
    """测试搜索功能"""
    print("🔍 开始测试搜索功能...")
    
    # 初始化系统
    config = Config()
    rag_system = RAGSystem(config)
    rag_system.initialize()
    
    # 加载数据
    data_file = "./data/xi_you_ji.txt"
    rag_system.load_and_index_data(data_file, max_documents=20, force_reload=True)
    
    # 查看已加载的文档
    print(f"\n📚 已加载文档数量: {len(rag_system.documents)}")
    print("前5个文档:")
    for i, doc in enumerate(rag_system.documents[:5]):
        print(f"  {i+1}: {doc[:100]}...")
    
    # 测试搜索
    test_queries = [
        "博物馆",
        "文化",
        "发酵床",
        "黄山",
        "京城"
    ]
    
    print(f"\n🔍 测试搜索查询:")
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = rag_system.search(query, top_k=3)
        
        if results:
            print(f"  找到 {len(results)} 个结果:")
            for i, result in enumerate(results):
                print(f"    {i+1}. 相似度: {result['score']:.4f}")
                print(f"       内容: {result['content'][:80]}...")
        else:
            print("  ❌ 没有找到结果")

if __name__ == "__main__":
    test_search() 