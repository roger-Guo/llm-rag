#!/usr/bin/env python3
"""
交互式RAG系统调试脚本
"""
import sys
import os
sys.path.append('./src')

def setup_debug_environment():
    """设置调试环境"""
    print("🔧 设置调试环境...")
    
    # 设置详细日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        print(f"✅ API密钥已设置: {api_key[:10]}...")
    else:
        print("⚠️ API密钥未设置")
    
    return True

def create_debug_session():
    """创建调试会话"""
    print("🚀 创建RAG系统调试会话...")
    
    try:
        from config import Config
        from rag_system import RAGSystem
        
        # 创建配置
        config = Config()
        print("✅ 配置加载成功")
        
        # 创建RAG系统
        rag_system = RAGSystem(config)
        print("✅ RAG系统创建成功")
        
        # 初始化系统
        print("🔄 正在初始化系统...")
        if rag_system.initialize():
            print("✅ 系统初始化成功")
            
            # 显示系统状态
            print(f"   - 使用ModelScope: {rag_system.using_modelscope}")
            print(f"   - 使用TF-IDF: {rag_system.using_tfidf}")
            print(f"   - 嵌入模型: {getattr(rag_system.embedding_model, '__class__.__name__', 'None')}")
            
            return rag_system
        else:
            print("❌ 系统初始化失败")
            return None
            
    except Exception as e:
        print(f"❌ 创建调试会话失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_debug(rag_system):
    """交互式调试"""
    print("\n🎯 进入交互式调试模式")
    print("可用命令:")
    print("  - search <query>: 搜索文档")
    print("  - query <question>: 完整问答")
    print("  - load <file> [max_docs]: 加载数据")
    print("  - stats: 显示统计信息")
    print("  - help: 显示帮助")
    print("  - quit: 退出")
    
    while True:
        try:
            command = input("\n🔍 调试命令> ").strip()
            
            if not command:
                continue
                
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("👋 退出调试模式")
                break
                
            elif cmd == 'help':
                print("可用命令:")
                print("  search <query> - 搜索文档")
                print("  query <question> - 完整问答")
                print("  load <file> [max_docs] - 加载数据")
                print("  stats - 显示统计信息")
                print("  help - 显示帮助")
                print("  quit - 退出")
                
            elif cmd == 'search' and len(parts) > 1:
                query = ' '.join(parts[1:])
                print(f"🔍 搜索: {query}")
                
                results = rag_system.search(query, top_k=3)
                print(f"找到 {len(results)} 个结果:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. 相似度: {result['score']:.3f}")
                    print(f"     内容: {result['content'][:100]}...")
                    
            elif cmd == 'query' and len(parts) > 1:
                question = ' '.join(parts[1:])
                print(f"🤖 问题: {question}")
                
                result = rag_system.query(question, top_k=3)
                print(f"⏱️  响应时间: {result['total_time']:.2f}s")
                print(f"📝 回答: {result['answer']}")
                
            elif cmd == 'load' and len(parts) > 1:
                file_path = parts[1]
                max_docs = int(parts[2]) if len(parts) > 2 else 100
                
                print(f"📚 加载数据: {file_path} (最多{max_docs}条)")
                success = rag_system.load_and_index_data(file_path, max_docs, force_reload=True)
                
                if success:
                    print("✅ 数据加载成功")
                    stats = rag_system.get_collection_stats()
                    print(f"📊 统计信息: {stats}")
                else:
                    print("❌ 数据加载失败")
                    
            elif cmd == 'stats':
                stats = rag_system.get_collection_stats()
                print("📊 系统统计信息:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                    
            else:
                print("❌ 未知命令，输入 'help' 查看可用命令")
                
        except KeyboardInterrupt:
            print("\n👋 退出调试模式")
            break
        except Exception as e:
            print(f"❌ 命令执行错误: {e}")

def main():
    """主函数"""
    print("🔍 RAG系统交互式调试器")
    print("=" * 50)
    
    # 设置环境
    if not setup_debug_environment():
        return
    
    # 创建调试会话
    rag_system = create_debug_session()
    if not rag_system:
        return
    
    # 进入交互模式
    interactive_debug(rag_system)

if __name__ == "__main__":
    main() 