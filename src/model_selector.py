"""
嵌入模型选择和管理工具
"""
import streamlit as st
from typing import Dict, List
import os
import shutil
from pathlib import Path

class ModelSelector:
    """模型选择器"""
    
    SMALL_MODELS = {
        "paraphrase-MiniLM-L3-v2": {
            "name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "size": "17MB",
            "description": "最小模型，速度最快",
            "languages": ["英文", "多语言"],
            "best_for": "资源受限环境"
        },
        "all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2", 
            "size": "22MB",
            "description": "平衡性能和大小的推荐选择",
            "languages": ["英文", "多语言"],
            "best_for": "通用问答系统"
        },
        "paraphrase-MiniLM-L6-v2": {
            "name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "size": "22MB", 
            "description": "针对相似句检测优化",
            "languages": ["英文", "多语言"],
            "best_for": "语义相似度检索"
        },
        "all-MiniLM-L12-v2": {
            "name": "sentence-transformers/all-MiniLM-L12-v2",
            "size": "33MB",
            "description": "更好效果，稍大一些",
            "languages": ["英文", "多语言"], 
            "best_for": "对效果要求较高的应用"
        }
    }
    
    @staticmethod
    def display_model_selector():
        """显示模型选择器界面"""
        st.subheader("🎯 嵌入模型选择")
        
        # 当前模型信息
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            current_model = ModelSelector._get_current_model()
            if current_model:
                st.info(f"📱 当前模型: {current_model}")
        
        # TF-IDF模式开关
        st.markdown("### ⚡ 快速模式")
        use_tfidf = st.checkbox(
            "🎯 使用TF-IDF模式 (无需下载模型，启动最快)", 
            value=True,
            help="TF-IDF模式使用传统文本相似度算法，无需下载大型模型，启动速度快"
        )
        
        if use_tfidf:
            st.success("✅ 当前配置: TF-IDF模式 - 快速启动，无需下载")
            if st.button("🚀 应用TF-IDF配置"):
                ModelSelector._switch_to_tfidf()
        else:
            # 嵌入模型选择
            st.markdown("### 📋 小型嵌入模型 (≤50MB)")
            
            # 使用选择框而不是expander
            model_options = list(ModelSelector.SMALL_MODELS.keys())
            selected_model = st.selectbox(
                "选择嵌入模型:",
                model_options,
                index=1,  # 默认选择all-MiniLM-L6-v2
                format_func=lambda x: f"{x} ({ModelSelector.SMALL_MODELS[x]['size']})"
            )
            
            if selected_model:
                model_info = ModelSelector.SMALL_MODELS[selected_model]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**描述**: {model_info['description']}")
                    st.markdown(f"**适用场景**: {model_info['best_for']}")
                    st.markdown(f"**支持语言**: {', '.join(model_info['languages'])}")
                    
                with col2:
                    st.metric("模型大小", model_info['size'])
                
                if st.button(f"🔄 切换到 {selected_model}"):
                    ModelSelector._switch_model(model_info['name'])
        
        # 模型管理工具
        st.markdown("### 🛠️ 模型管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ 清理模型缓存"):
                ModelSelector._clear_model_cache()
        
        with col2:
            if st.button("📊 检查磁盘使用"):
                ModelSelector._check_disk_usage()
        
        with col3:
            if st.button("🔍 扫描已下载模型"):
                ModelSelector._scan_downloaded_models()
    
    @staticmethod
    def _get_current_model() -> str:
        """获取当前使用的模型"""
        try:
            if hasattr(st.session_state.rag_system, 'embedding_model'):
                if st.session_state.rag_system.using_modelscope:
                    return "ModelScope中文模型"
                elif st.session_state.rag_system.using_tfidf:
                    return "TF-IDF (备选方案)"
                else:
                    # 尝试从模型路径获取名称
                    return "Sentence Transformer模型"
            return "未知"
        except:
            return "未初始化"
    
    @staticmethod 
    def _switch_model(model_name: str):
        """切换模型"""
        try:
            with st.spinner(f"正在切换到模型: {model_name}"):
                # 更新配置
                if 'config' in st.session_state:
                    st.session_state.config.EMBEDDING_MODEL_NAME = model_name
                
                # 重置RAG系统
                st.session_state.rag_system = None
                st.session_state.system_initialized = False
                
                st.success(f"✅ 已切换到模型: {model_name}")
                st.info("💡 请重新初始化系统以使用新模型")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ 切换模型失败: {e}")
    
    @staticmethod
    def _clear_model_cache():
        """清理模型缓存"""
        try:
            models_dir = Path("../models")
            if models_dir.exists():
                # 计算缓存大小
                total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
                size_mb = total_size / 1024 / 1024
                
                # 清理缓存
                shutil.rmtree(models_dir)
                models_dir.mkdir(exist_ok=True)
                
                st.success(f"✅ 已清理模型缓存 (释放 {size_mb:.1f}MB)")
            else:
                st.info("📁 模型缓存目录为空")
                
        except Exception as e:
            st.error(f"❌ 清理缓存失败: {e}")
    
    @staticmethod
    def _check_disk_usage():
        """检查磁盘使用情况"""
        try:
            models_dir = Path("../models")
            chroma_dir = Path("../chroma_db")
            
            usage_info = []
            
            for dir_path, name in [(models_dir, "模型缓存"), (chroma_dir, "向量数据库")]:
                if dir_path.exists():
                    total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    size_mb = total_size / 1024 / 1024
                    usage_info.append(f"📁 {name}: {size_mb:.1f}MB")
                else:
                    usage_info.append(f"📁 {name}: 0MB")
            
            st.info("\n".join(usage_info))
            
        except Exception as e:
            st.error(f"❌ 检查磁盘使用失败: {e}")
    
    @staticmethod
    def _scan_downloaded_models():
        """扫描已下载的模型"""
        try:
            models_dir = Path("../models")
            
            if not models_dir.exists():
                st.info("📁 模型目录不存在")
                return
            
            downloaded_models = []
            
            for item in models_dir.iterdir():
                if item.is_dir():
                    # 计算目录大小
                    total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    size_mb = total_size / 1024 / 1024
                    downloaded_models.append(f"📦 {item.name}: {size_mb:.1f}MB")
            
            if downloaded_models:
                st.info("已下载的模型:\n" + "\n".join(downloaded_models))
            else:
                st.info("📭 暂无已下载的模型")
                
        except Exception as e:
            st.error(f"❌ 扫描模型失败: {e}")

    @staticmethod
    def _switch_to_tfidf():
        """切换到TF-IDF模式"""
        try:
            with st.spinner("正在切换到TF-IDF模式..."):
                # 更新配置
                if 'config' in st.session_state:
                    st.session_state.config.USE_TFIDF_ONLY = True
                
                # 重置RAG系统
                st.session_state.rag_system = None
                st.session_state.system_initialized = False
                
                st.success("✅ 已切换到TF-IDF模式")
                st.info("💡 请重新初始化系统以使用TF-IDF")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ 切换TF-IDF模式失败: {e}")

# 在app.py中集成模型选择器
def integrate_model_selector():
    """集成模型选择器到主应用"""
    from model_selector import ModelSelector
    
    with st.sidebar:
        with st.expander("🎯 模型选择", expanded=False):
            ModelSelector.display_model_selector() 