"""
RAG (Retrieval-Augmented Generation) 系统
集成 ModelScope、ChromaDB、DeepSeek API
"""
import os
import sys
import time
import json
import warnings
from typing import List, Dict, Any, Optional
import logging

# 抑制警告
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # ModelScope镜像源配置
    os.environ['MODELSCOPE_CACHE'] = '../models'
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi
    from sentence_transformers import SentenceTransformer
    MODELSCOPE_AVAILABLE = True
    logger.info("✅ ModelScope available")
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logger.warning("⚠️ ModelScope not available")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB available")
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("⚠️ ChromaDB not available")

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("✅ OpenAI client available")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI client not available")

from utils import load_toutiao_data, clean_text, split_text_by_sentences


class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, config):
        """
        初始化RAG系统
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.openai_client = None
        self.using_modelscope = False
        
        # 确保目录存在
        os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(self.config.CHROMA_PERSIST_DIR, exist_ok=True)
        
        logger.info("🚀 RAG系统初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            bool: 初始化是否成功
        """
        logger.info("🔧 开始初始化RAG系统...")
        
        # 初始化嵌入模型
        if not self._initialize_embedding_model():
            logger.error("❌ 嵌入模型初始化失败")
            return False
        
        # 初始化向量数据库
        if not self._initialize_vector_db():
            logger.error("❌ 向量数据库初始化失败")
            return False
        
        # 初始化DeepSeek客户端（可选）
        self._initialize_openai_client()
        
        logger.info("✅ RAG系统初始化成功")
        return True
    
    def _initialize_embedding_model(self) -> bool:
        """初始化嵌入模型"""
        logger.info("🤖 正在初始化嵌入模型...")
        
        # 🎯 优先尝试使用配置的M3E-Base模型
        if self.config.EMBEDDING_MODEL_NAME == "AI-ModelScope/m3e-base":
            try:
                logger.info(f"🚀 优先尝试加载M3E-Base模型: {self.config.EMBEDDING_MODEL_NAME}")
                
                # 尝试两个可能的路径：配置路径和当前目录路径
                possible_paths = [
                    os.path.join(self.config.MODEL_CACHE_DIR, "AI-ModelScope", "m3e-base"),  # 配置路径
                    os.path.join("./models", "AI-ModelScope", "m3e-base"),  # 当前目录路径
                    os.path.join("models", "AI-ModelScope", "m3e-base")  # 相对路径
                ]
                
                for local_model_path in possible_paths:
                    if os.path.exists(local_model_path):
                        logger.info(f"📁 找到本地模型路径: {local_model_path}")
                        # 使用本地路径加载模型
                        self.embedding_model = SentenceTransformer(local_model_path)
                        logger.info(f"✅ M3E-Base模型加载成功: {local_model_path}")
                        return True
                
                logger.warning("⚠️ 未找到本地M3E-Base模型文件")
                    
            except Exception as e:
                logger.warning(f"⚠️ M3E-Base模型加载失败: {e}")
        
        # 尝试直接加载其他配置的嵌入模型
        try:
            logger.info(f"🚀 尝试直接加载配置的嵌入模型: {self.config.EMBEDDING_MODEL_NAME}")
            
            # 尝试加载配置的嵌入模型
            self.embedding_model = SentenceTransformer(
                self.config.EMBEDDING_MODEL_NAME,
                cache_folder=self.config.MODEL_CACHE_DIR
            )
            
            logger.info(f"✅ 嵌入模型加载成功: {self.config.EMBEDDING_MODEL_NAME}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 配置的嵌入模型直接加载失败: {e}")
        
        # 尝试使用ModelScope（如果可用且网络条件好）
        if MODELSCOPE_AVAILABLE:
            try:
                logger.info("🚀 尝试使用ModelScope镜像源...")
                
                # 中文嵌入模型
                model_id = 'iic/nlp_gte_sentence-embedding_chinese-small'
                
                # 下载模型
                model_dir = snapshot_download(model_id, cache_dir=self.config.MODEL_CACHE_DIR)
                logger.info(f"✅ 模型下载成功: {model_dir}")
                
                # 加载模型
                self.embedding_model = SentenceTransformer(model_dir)
                self.using_modelscope = True
                logger.info("✅ ModelScope嵌入模型加载成功")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ ModelScope加载失败: {e}")
        
        logger.error("❌ 所有嵌入模型加载失败")
        return False
    
    def _initialize_vector_db(self) -> bool:
        """初始化向量数据库"""
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB不可用")
            return False
        
        try:
            logger.info("🗄️ 正在初始化ChromaDB...")
            
            # 创建ChromaDB客户端
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.config.COLLECTION_NAME
                )
                logger.info(f"✅ 已连接到现有集合: {self.config.COLLECTION_NAME}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"✅ 创建新集合: {self.config.COLLECTION_NAME}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ChromaDB初始化失败: {e}")
            return False
    
    def _initialize_openai_client(self):
        """初始化OpenAI客户端（可选）"""
        if not OPENAI_AVAILABLE:
            logger.warning("⚠️ OpenAI客户端不可用")
            return
        
        # 安全地显示API密钥（只显示前10个字符）
        api_key_display = self.config.DEEPSEEK_API_KEY[:10] + "..." if self.config.DEEPSEEK_API_KEY else "未设置"
        logger.info(f"✅ DeepSeek API密钥: {api_key_display}")
        
        try:
            if self.config.DEEPSEEK_API_KEY and self.config.DEEPSEEK_API_KEY != "sk-YOUR-API-KEY":
                self.openai_client = openai.OpenAI(
                    api_key=self.config.DEEPSEEK_API_KEY,
                    base_url=self.config.DEEPSEEK_BASE_URL
                )
                logger.info("✅ DeepSeek客户端初始化成功")
            else:
                logger.warning("⚠️ DeepSeek API密钥未配置")
        except Exception as e:
            logger.warning(f"⚠️ DeepSeek客户端初始化失败: {e}")
    
    def load_and_index_data(self, data_file: str, max_documents: int = 1000, force_reload: bool = False) -> bool:
        """
        加载和索引数据
        
        Args:
            data_file: 数据文件路径
            max_documents: 最大文档数量
            force_reload: 是否强制重新加载
        
        Returns:
            bool: 是否成功
        """
        logger.info(f"📚 开始加载数据: {data_file}")
        
        # 检查是否需要重新加载
        if not force_reload and self.collection:
            try:
                count = self.collection.count()
                if count > 0:
                    logger.info(f"✅ 数据已存在 ({count} 条文档)")
                    return True
            except:
                pass
        
        # 清空现有数据（如果强制重新加载）
        if force_reload and self.collection:
            try:
                self.chroma_client.delete_collection(self.config.COLLECTION_NAME)
                self.collection = self.chroma_client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("🗑️ 已清空现有数据")
            except Exception as e:
                logger.warning(f"⚠️ 清空数据失败: {e}")
        
        # 加载原始数据
        raw_data = load_toutiao_data(data_file, max_documents)
        if not raw_data:
            logger.error("❌ 数据加载失败")
            return False
        
        logger.info(f"📖 加载了 {len(raw_data)} 条原始数据")
        
        # 处理和分块数据
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, item in enumerate(raw_data):
            # 清理文本
            content = clean_text(item.get('content', ''))
            title = clean_text(item.get('title', ''))
            
            if not content:
                continue
            
            # 分块处理
            chunks = split_text_by_sentences(
                content, 
                self.config.MAX_CHUNK_SIZE, 
                self.config.CHUNK_OVERLAP
            )
            
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:  # 跳过太短的块
                    continue
                
                chunk_id = f"doc_{i}_chunk_{j}"
                logger.info("✅ 文本块: "+chunk[0:100])
                all_chunks.append(chunk)
                all_metadatas.append({
                    'title': title,
                    'category': item.get('category', ''),
                    'keywords': item.get('keywords', ''),
                    'doc_id': i,
                    'chunk_id': j
                })
                all_ids.append(chunk_id)
        
        if not all_chunks:
            logger.error("❌ 没有有效的文本块")
            return False
        
        logger.info(f"📝 生成了 {len(all_chunks)} 个文本块")
        
        # 索引数据
        return self._index_chunks(all_chunks, all_metadatas, all_ids)
    
    def _index_chunks(self, chunks: List[str], metadatas: List[Dict], ids: List[str]) -> bool:
        """索引文本块"""
        try:
            # 使用嵌入模型
            logger.info("🧮 正在生成嵌入向量...")
            
            # 批量生成嵌入向量
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_chunks,
                    convert_to_tensor=False,
                    show_progress_bar=True
                )
                embeddings.extend(batch_embeddings.tolist())
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"📊 已处理 {min(i + batch_size, len(chunks))}/{len(chunks)} 个文本块")
            
            # 存储到ChromaDB
            logger.info("💾 正在存储到向量数据库...")
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info("✅ 向量索引完成")
            return True
                
        except Exception as e:
            logger.error(f"❌ 索引失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            List[Dict]: 搜索结果
        """
        try:
            return self._search_embedding(query, top_k)
        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []
    
    def _search_embedding(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """使用嵌入向量搜索"""
        if not self.collection:
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],  # 转换为相似度
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
            })
        
        return formatted_results
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        使用DeepSeek生成答案
        
        Args:
            query: 用户问题
            context: 检索到的上下文
        
        Returns:
            str: 生成的答案
        """
        if not self.openai_client:
            return f"基于检索到的信息：\n{context[:500]}..."
        
        try:
            prompt = f"""
基于以下信息回答用户问题。请确保答案准确、简洁且有用。

上下文信息：
{context}

用户问题：{query}

请根据上下文信息回答问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。
"""
            
            response = self.openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个有用的AI助手，能够基于提供的信息准确回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ 答案生成失败: {e}")
            return f"抱歉，无法生成回答。基于检索信息：\n{context[:500]}..."
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        完整的问答流程
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
        
        Returns:
            Dict: 包含答案、来源和性能指标的结果
        """
        start_time = time.time()
        
        # 搜索相关文档
        search_start = time.time()
        sources = self.search(question, top_k)
        search_time = time.time() - search_start
        
        # 构建上下文
        context = "\n\n".join([
            f"相关信息 {i+1}：{source['content']}"
            for i, source in enumerate(sources[:3])  # 只用前3个结果
        ])
        logger.info("✅ 上下文: "+context)
        # 生成答案
        generate_start = time.time()
        answer = self.generate_answer(question, context)
        generate_time = time.time() - generate_start
        
        total_time = time.time() - start_time
        
        return {
            'answer': answer,
            'sources': sources,
            'search_time': search_time,
            'generate_time': generate_time,
            'total_time': total_time,
            'using_modelscope': self.using_modelscope
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if self.collection:
                return {
                    'total_documents': self.collection.count(),
                    'embedding_model': self.config.EMBEDDING_MODEL_NAME,
                    'using_modelscope': self.using_modelscope,
                    'chunk_size': self.config.MAX_CHUNK_SIZE,
                    'chunk_overlap': self.config.CHUNK_OVERLAP,
                    'collection_name': self.config.COLLECTION_NAME
                }
            else:
                return {'error': '系统未初始化'}
        except Exception as e:
            return {'error': str(e)} 