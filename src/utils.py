"""
RAG系统工具函数
"""
import json
import re
from typing import List, Dict, Any


def load_toutiao_data(file_path: str, max_lines: int = 10000) -> List[Dict[str, Any]]:
    """
    加载头条新闻数据
    
    Args:
        file_path: 数据文件路径
        max_lines: 最大加载行数
    
    Returns:
        新闻数据列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 尝试JSON格式
                    item = json.loads(line)
                    if 'text' in item and 'category' in item:
                        # 标准化数据格式
                        standardized_item = {
                            'id': f"news_{i}",
                            'title': item.get('title', ''),
                            'content': item.get('text', ''),
                            'category': item.get('category', ''),
                            'keywords': item.get('keywords', '')
                        }
                        data.append(standardized_item)
                except json.JSONDecodeError:
                    # 尝试分隔符格式 (id_!_code_!_category_!_title_!_keywords)
                    try:
                        parts = line.split('_!_')
                        if len(parts) >= 4:
                            news_id = parts[0]
                            code = parts[1] if len(parts) > 1 else ''
                            category = parts[2] if len(parts) > 2 else ''
                            title = parts[3] if len(parts) > 3 else ''
                            keywords = parts[4] if len(parts) > 4 else ''
                            
                            # 清理类别名称
                            if category.startswith('news_'):
                                category = category[5:]  # 移除 'news_' 前缀
                            
                            standardized_item = {
                                'id': f"news_{i}",
                                'title': title,
                                'content': title,  # 使用标题作为内容
                                'category': category,
                                'keywords': keywords
                            }
                            data.append(standardized_item)
                    except Exception as e:
                        print(f"解析行失败 {i}: {e}")
                        continue
                        
    except FileNotFoundError:
        print(f"数据文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []
    
    print(f"成功加载 {len(data)} 条数据")
    return data


def clean_text(text: str) -> str:
    """
    清理文本数据
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留中文、英文、数字、常见标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】\-\s]', '', text)
    
    # 移除过短的文本
    if len(text.strip()) < 10:
        return ""
    
    return text.strip()


def split_text_by_sentences(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    按句子分割文本
    
    Args:
        text: 输入文本
        max_chunk_size: 最大块大小
        overlap: 重叠大小
    
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    # 按句子分割
    sentences = re.split(r'[。！？；]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前块加上新句子超过最大长度
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # 重叠处理：保留最后一部分作为下一块的开始
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += sentence + "。"
    
    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    格式化搜索结果用于显示
    
    Args:
        results: 搜索结果列表
    
    Returns:
        格式化的字符串
    """
    if not results:
        return "没有找到相关结果。"
    
    formatted = []
    for i, result in enumerate(results, 1):
        content = result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
        score = result.get('score', 0)
        metadata = result.get('metadata', {})
        
        formatted.append(f"""
**结果 {i}** (相似度: {score:.3f})
- **类别**: {metadata.get('category', '未知')}
- **标题**: {metadata.get('title', '无标题')}
- **内容**: {content}
        """)
    
    return "\n".join(formatted) 