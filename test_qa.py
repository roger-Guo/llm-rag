#!/usr/bin/env python3
"""
快速问答测试脚本
"""
import sys
import os
sys.path.append('./src')

from config import Config
from rag_system import RAGSystem

def test_qa():
    """测试问答功能"""
    print("🚀 开始测试RAG问答功能...")
    
    # 初始化系统
    config = Config()
    rag_system = RAGSystem(config)
    
    print("📊 正在初始化系统...")
    if not rag_system.initialize():
        print("❌ 系统初始化失败")
        return
    
    # 加载少量数据进行测试
    print("📚 正在加载测试数据...")
    data_file = "./data/三国演义.txt"
    success = rag_system.load_and_index_data(data_file, max_documents=50, force_reload=True)
    
    if not success:
        print("❌ 数据加载失败")
        return
    
    print("✅ 系统准备完成！")
    
    # 测试问题
    test_questions = [
        "什么是博物馆？",
        "文化旅游有什么推荐？",
        "黄山黄河相关的内容",
        "发酵床是什么？"
    ]
    
    print("\n🤖 开始问答测试...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"🔍 问题 {i}: {question}")
        
        # 执行查询
        result = rag_system.query(question, top_k=3)
        
        print(f"⏱️  响应时间: {result['total_time']:.2f}s")
        print(f"📝 回答: {result['answer'][:200]}...")
        
        if result['sources']:
            print(f"📚 找到 {len(result['sources'])} 个相关源")
            for j, source in enumerate(result['sources'][:2]):
                print(f"   源 {j+1}: {source['content'][:100]}... (相似度: {source['score']:.3f})")
        
        print("-" * 50)
    
    print("🎉 测试完成！")

if __name__ == "__main__":
    test_qa() 