"""
RAG智能问答系统 - Streamlit Web界面
基于ModelScope镜像源的RAG系统
"""
import streamlit as st
import os
import sys
import time
import traceback
from pathlib import Path

# 设置页面配置 - 必须在其他streamlit命令之前
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
    from rag_system import RAGSystem
    from utils import format_search_results
    from model_selector import ModelSelector
except ImportError as e:
    st.error(f"❌ 导入模块失败: {e}")
    st.stop()

# 初始化session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'config' not in st.session_state:
    st.session_state.config = None

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统（带缓存）"""
    try:
        config = Config()
        rag_system = RAGSystem(config)
        return rag_system, config
    except Exception as e:
        st.error(f"❌ 初始化配置失败: {e}")
        return None, None

def create_sample_data():
    """创建示例数据文件"""
    try:
        data_dir = Path("../data")
        data_dir.mkdir(exist_ok=True)
        
        sample_file = data_dir / "三国演义.txt"
        
        if not sample_file.exists():
            sample_data = [
                '{"text": "苹果公司发布了最新的iPhone 15系列手机，采用了全新的A17芯片，性能提升显著。新手机支持USB-C接口，取代了传统的Lightning接口。", "category": "科技", "title": "苹果发布iPhone 15", "keywords": "苹果,iPhone,科技"}',
                '{"text": "中国足球队在亚洲杯预选赛中以3:1的比分战胜了对手，成功晋级下一轮。这是中国队近年来在国际赛场上的重要胜利。", "category": "体育", "title": "中国足球队获胜", "keywords": "足球,体育,亚洲杯"}',
                '{"text": "最新研究表明，地中海饮食有助于降低心血管疾病的风险。研究人员建议多食用橄榄油、坚果和鱼类等健康食品。", "category": "健康", "title": "地中海饮食有益健康", "keywords": "健康,饮食,心血管"}',
                '{"text": "电动汽车市场持续增长，特斯拉、比亚迪等品牌在全球销量创新高。专家预测，2024年电动汽车将占据更大的市场份额。", "category": "汽车", "title": "电动汽车市场增长", "keywords": "电动汽车,特斯拉,比亚迪"}',
                '{"text": "人工智能技术在医疗领域的应用越来越广泛，从诊断辅助到药物研发，AI正在改变传统医疗模式。", "category": "科技", "title": "AI在医疗领域应用", "keywords": "人工智能,医疗,科技"}',
                '{"text": "全球气候变化问题日益严重，各国正在加强合作应对碳排放。联合国气候峰会讨论了新的减排目标和政策。", "category": "环境", "title": "气候变化应对", "keywords": "气候变化,环境,减排"}',
                '{"text": "在线教育平台用户数量激增，远程学习成为新趋势。疫情推动了教育数字化转型的进程。", "category": "教育", "title": "在线教育发展", "keywords": "在线教育,远程学习,数字化"}',
                '{"text": "房地产市场出现调整，一线城市房价有所回落。政府出台了一系列调控政策以稳定市场。", "category": "财经", "title": "房地产市场调整", "keywords": "房地产,房价,调控政策"}',
                '{"text": "量子计算研究取得重大突破，新型量子处理器的计算能力大幅提升，有望在密码学和材料科学等领域发挥重要作用。", "category": "科技", "title": "量子计算突破", "keywords": "量子计算,科技,处理器"}',
                '{"text": "旅游业逐步复苏，国内游和出境游需求都在增长。各地推出了丰富的旅游产品和优惠政策。", "category": "旅游", "title": "旅游业复苏", "keywords": "旅游,复苏,国内游"}'
            ]
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                for item in sample_data:
                    f.write(f"{item}\n")
            
            return str(sample_file)
        return str(sample_file)
    except Exception as e:
        st.error(f"❌ 创建示例数据失败: {e}")
        return None

def display_system_info():
    """显示系统信息"""
    with st.sidebar:
        st.header("📊 系统状态")
        
        if st.session_state.system_initialized and st.session_state.rag_system:
            try:
                stats = st.session_state.rag_system.get_collection_stats()
                
                if 'error' not in stats:
                    st.markdown('<div class="status-box success-box">✅ 系统运行正常</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("文档数量", stats.get('total_documents', 0))
                    with col2:
                        if stats.get('using_modelscope', False):
                            st.metric("嵌入模型", "ModelScope")
                        elif stats.get('using_tfidf', False):
                            st.metric("嵌入模型", "TF-IDF")
                        else:
                            st.metric("嵌入模型", "本地模型")
                    
                    if stats.get('using_modelscope', False):
                        st.markdown('<div class="status-box info-box">🚀 使用ModelScope镜像源</div>', unsafe_allow_html=True)
                    elif stats.get('using_tfidf', False):
                        st.markdown('<div class="status-box warning-box">⚠️ 使用TF-IDF备选方案</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-box error-box">❌ 系统状态异常</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown('<div class="status-box error-box">❌ 获取状态失败</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box warning-box">⚠️ 系统未初始化</div>', unsafe_allow_html=True)

def main():
    """主应用函数"""
    
    # 标题
    st.markdown('<h1 class="main-header">🤖 RAG智能问答系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">基于ModelScope镜像源 + ChromaDB + DeepSeek的智能问答系统</p>', unsafe_allow_html=True)
    
    # 显示系统信息
    display_system_info()
    
    # 侧边栏控制
    with st.sidebar:
        st.header("🛠️ 系统控制")
        
        # 初始化按钮
        if not st.session_state.system_initialized:
            if st.button("🚀 初始化系统", type="primary"):
                with st.spinner("正在初始化RAG系统..."):
                    try:
                        rag_system, config = initialize_rag_system()
                        if rag_system and config:
                            st.session_state.rag_system = rag_system
                            st.session_state.config = config
                            success = rag_system.initialize()
                            if success:
                                st.session_state.system_initialized = True
                                st.success("✅ 系统初始化成功！")
                                st.rerun()
                            else:
                                st.error("❌ 系统初始化失败")
                        else:
                            st.error("❌ 配置加载失败")
                    except Exception as e:
                        st.error(f"❌ 初始化异常: {e}")
                        st.text(traceback.format_exc())
        
        # 数据管理（仅在系统初始化后显示）
        if st.session_state.system_initialized:
            st.header("📁 数据管理")
            
            # 生成示例数据
            if st.button("🎯 生成示例数据"):
                with st.spinner("正在生成示例数据..."):
                    sample_file = create_sample_data()
                    if sample_file:
                        st.success(f"✅ 示例数据已生成")
                    else:
                        st.error("❌ 生成示例数据失败")
            
            # 加载数据
            max_docs = st.slider("最大文档数量", 10, 1000, 100, 10)
            force_reload = st.checkbox("强制重新加载")
            
            if st.button("📚 加载数据到向量库"):
                data_file = "../data/三国演义.txt"
                if not os.path.exists(data_file):
                    create_sample_data()
                
                with st.spinner("正在加载和索引数据..."):
                    try:
                        success = st.session_state.rag_system.load_and_index_data(
                            data_file, 
                            max_documents=max_docs,
                            force_reload=force_reload
                        )
                        if success:
                            st.success("✅ 数据加载成功！")
                            st.rerun()
                        else:
                            st.error("❌ 数据加载失败")
                    except Exception as e:
                        st.error(f"❌ 加载数据异常: {e}")
        
            # 模型选择器
            st.header("�� 模型选择")
            ModelSelector.display_model_selector()
    
    # 主界面
    if st.session_state.system_initialized:
        display_main_interface()
    else:
        st.info("👆 请先在侧边栏初始化系统")
        
        # 显示帮助信息
        st.markdown("""
        ## 🚀 快速开始
        
        1. **初始化系统**: 点击侧边栏的"🚀 初始化系统"按钮
        2. **生成数据**: 点击"🎯 生成示例数据"创建测试数据  
        3. **加载数据**: 点击"📚 加载数据到向量库"建立索引
        4. **开始提问**: 在问答界面输入问题
        
        ## 💡 提示
        - 首次运行会下载中文嵌入模型（约512MB）
        - 如果网络问题，系统会自动降级到TF-IDF方案
        - 支持中文问答，效果更佳
        """)

def display_main_interface():
    """显示主界面"""
    # 创建标签页
    tab1, tab2 = st.tabs(["💬 智能问答", "🔍 系统监控"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_monitoring_interface()

def display_chat_interface():
    """智能问答界面"""
    st.header("💬 智能问答")
    
    # 检查系统状态
    if not st.session_state.rag_system:
        st.warning("⚠️ RAG系统未初始化")
        return
    
    # 查询设置
    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.selectbox("检索结果数量", [3, 5, 8, 10], index=1)
    
    # 问题输入
    question = st.text_input(
        "请输入您的问题:",
        placeholder="例如: 苹果公司最新发布了什么产品？",
        help="输入任何问题，系统会基于知识库为您提供答案"
    )
    
    # 预设问题
    st.subheader("🔥 热门问题")
    preset_questions = [
        "苹果公司最新发布了什么产品？",
        "电动汽车市场发展如何？",
        "人工智能在医疗领域有什么应用？",
        "气候变化有什么应对措施？",
        "在线教育发展怎么样？"
    ]
    
    # 使用按钮组显示预设问题
    cols = st.columns(len(preset_questions))
    for i, q in enumerate(preset_questions):
        with cols[i % len(cols)]:
            if st.button(f"📝 {q[:8]}...", key=f"preset_{i}", help=q):
                st.session_state.current_question = q
                st.rerun()
    
    # 使用session state中的问题（如果有）
    if 'current_question' in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    # 处理问答
    if question:
        try:
            with st.spinner("🔍 正在搜索相关信息..."):
                result = st.session_state.rag_system.query(question, top_k=top_k)
                
                # 显示答案
                st.subheader("🎯 智能回答")
                st.markdown(f"**问题:** {question}")
                
                if result['answer']:
                    st.markdown(f"**回答:** {result['answer']}")
                else:
                    st.warning("抱歉，没有找到相关信息。")
                
                # 显示性能指标
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("检索时间", f"{result['search_time']:.2f}s")
                with col_b:
                    st.metric("生成时间", f"{result['generate_time']:.2f}s")
                with col_c:
                    st.metric("总时间", f"{result['total_time']:.2f}s")
                
                # 显示参考来源
                if result.get('sources'):
                    st.subheader("📚 参考来源")
                    
                    for i, source in enumerate(result['sources'][:top_k]):
                        with st.expander(f"来源 {i+1} (相似度: {source['score']:.3f})"):
                            metadata = source.get('metadata', {})
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                if metadata.get('title'):
                                    st.markdown(f"**标题:** {metadata['title']}")
                                if metadata.get('category'):
                                    st.markdown(f"**类别:** {metadata['category']}")
                                if metadata.get('keywords'):
                                    st.markdown(f"**关键词:** {metadata['keywords']}")
                            with col2:
                                st.metric("相似度", f"{source['score']:.3f}")
                            
                            st.markdown("**内容:**")
                            st.text(source.get('content', ''))
        except Exception as e:
            st.error(f"❌ 查询失败: {e}")
            st.text(traceback.format_exc())

def display_monitoring_interface():
    """系统监控界面"""
    st.header("📈 系统监控")
    
    try:
        if st.session_state.rag_system:
            stats = st.session_state.rag_system.get_collection_stats()
            
            # 系统状态概览
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("文档总数", stats.get('total_documents', 0))
            with col2:
                if stats.get('using_modelscope'):
                    st.metric("嵌入方案", "ModelScope", "🚀")
                elif stats.get('using_tfidf'):
                    st.metric("嵌入方案", "TF-IDF", "⚠️")
                else:
                    st.metric("嵌入方案", "本地模型", "💻")
            with col3:
                st.metric("分块大小", stats.get('chunk_size', 'N/A'))
            with col4:
                st.metric("重叠大小", stats.get('chunk_overlap', 'N/A'))
            
            # 配置信息
            st.subheader("⚙️ 系统配置")
            config_data = {
                "嵌入模型": stats.get('embedding_model', 'N/A'),
                "集合名称": stats.get('collection_name', 'N/A'),
                "最大分块大小": stats.get('chunk_size', 'N/A'),
                "分块重叠": stats.get('chunk_overlap', 'N/A'),
                "使用ModelScope": "是" if stats.get('using_modelscope') else "否",
                "使用TF-IDF": "是" if stats.get('using_tfidf') else "否"
            }
            
            for key, value in config_data.items():
                st.text(f"{key}: {value}")
            
            # 操作按钮
            st.subheader("🛠️ 系统操作")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 重新初始化系统"):
                    st.session_state.rag_system = None
                    st.session_state.system_initialized = False
                    st.success("✅ 系统已重置，请重新初始化")
                    st.rerun()
            
            with col2:
                if st.button("📊 刷新状态"):
                    st.rerun()
        else:
            st.warning("⚠️ RAG系统未初始化")
    except Exception as e:
        st.error(f"❌ 获取系统信息失败: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ 应用运行异常: {e}")
        st.text(traceback.format_exc()) 