"""
RAG系统配置文件
"""
import os

# 环境变量名称
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

# DeepSeek API配置 - 从环境变量获取，如果没有则使用默认值
DEEPSEEK_API_KEY = os.getenv(DEEPSEEK_API_KEY_ENV, "")  # 从环境变量获取API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

class Config:
    """RAG系统配置类"""
    
    # 数据路径
    DATA_DIR = "../data"
    TOUTIAO_DATA_FILE = os.path.join(DATA_DIR, "三国演义.txt")
    
    # 嵌入模型配置
    MODEL_CACHE_DIR = "../models"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22MB 高效小型模型
    
    # 🎯 TF-IDF优先模式 - 设置为True直接使用TF-IDF，跳过嵌入模型下载
    USE_TFIDF_ONLY = True
    
    # 备选小型模型列表（按大小排序）
    ALTERNATIVE_MODELS = [
        "sentence-transformers/paraphrase-MiniLM-L3-v2",  # 17MB 最小
        "sentence-transformers/all-MiniLM-L6-v2",         # 22MB 推荐
        "sentence-transformers/paraphrase-MiniLM-L6-v2",  # 22MB 相似句优化
        "sentence-transformers/all-MiniLM-L12-v2"         # 33MB 更好效果
    ]
    
    # ChromaDB配置
    CHROMA_PERSIST_DIR = "../chroma_db"
    COLLECTION_NAME = "toutiao_news"
    
    # 文本处理配置
    MAX_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
    DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL
    
    # 搜索配置
    DEFAULT_TOP_K = 5
    
    def __init__(self):
        """确保目录存在"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIR, exist_ok=True) 