"""
RAG系统工具函数
"""
import json
import re
from typing import List, Dict, Any
import string


def load_toutiao_data(file_path: str, max_lines: int = 10000) -> List[Dict[str, Any]]:
    """
    加载小说文本数据
    
    Args:
        file_path: 数据文件路径
        max_lines: 最大加载行数
    
    Returns:
        小说数据列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按章节分割小说
        chapters = split_novel_by_chapters(content)
        
        for i, chapter in enumerate(chapters):
            if i >= max_lines:
                break
            
            if not chapter.strip():
                continue
            
            # 提取章节标题和内容
            chapter_title, chapter_content = extract_chapter_info(chapter)
            
            # 标准化数据格式
            standardized_item = {
                'id': f"chapter_{i}",
                'title': chapter_title,
                'content': chapter_content,
                'category': '小说',
                'keywords': extract_keywords(chapter_content)
            }
            data.append(standardized_item)
                        
    except FileNotFoundError:
        print(f"数据文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []
    
    print(f"成功加载 {len(data)} 个章节")
    return data


def split_novel_by_chapters(content: str) -> List[str]:
    """
    按章节分割小说内容
    
    Args:
        content: 小说全文
    
    Returns:
        章节列表
    """
    # 常见的章节分割模式
    chapter_patterns = [
        r'第[一二三四五六七八九十百千万\d]+章.*?\n',  # 第X章
        r'第[一二三四五六七八九十百千万\d]+回.*?\n',  # 第X回
        r'Chapter\s*\d+.*?\n',  # Chapter X
        r'第[一二三四五六七八九十百千万\d]+节.*?\n',  # 第X节
        r'[一二三四五六七八九十百千万\d]+、.*?\n',    # 数字、标题
    ]
    
    chapters = []
    current_pos = 0
    
    # 查找所有章节标题位置
    chapter_positions = []
    for pattern in chapter_patterns:
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        chapter_positions.extend(matches)
    
    # 按位置排序
    chapter_positions.sort(key=lambda x: x.start())
    
    if not chapter_positions:
        # 如果没有找到章节标题，按段落分割
        paragraphs = content.split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50:  # 只保留较长的段落
                chapters.append(para.strip())
        return chapters
    
    # 分割章节
    for i, match in enumerate(chapter_positions):
        start = match.start()
        
        if i > 0:
            # 添加前一章节内容
            chapter_content = content[current_pos:start].strip()
            if chapter_content:
                chapters.append(chapter_content)
        
        current_pos = start
    
    # 添加最后一章
    if current_pos < len(content):
        last_chapter = content[current_pos:].strip()
        if last_chapter:
            chapters.append(last_chapter)
    
    return chapters


def extract_chapter_info(chapter: str) -> tuple:
    """
    提取章节标题和内容
    
    Args:
        chapter: 章节文本
    
    Returns:
        (标题, 内容) 元组
    """
    lines = chapter.split('\n')
    
    # 第一行作为标题
    title = lines[0].strip() if lines else "未知章节"
    
    # 移除标题行，其余作为内容
    content_lines = lines[1:] if len(lines) > 1 else lines
    content = '\n'.join(content_lines).strip()
    
    # 如果内容为空，使用标题作为内容
    if not content:
        content = title
    
    return title, content


def extract_keywords(content: str, max_keywords: int = 5) -> str:
    """
    从内容中提取关键词
    
    Args:
        content: 文本内容
        max_keywords: 最大关键词数量
    
    Returns:
        关键词字符串
    """
    # 简单的关键词提取：选择出现频率较高的词
    words = re.findall(r'[\u4e00-\u9fa5]{2,}', content)
    
    if not words:
        return ""
    
    # 统计词频
    word_freq = {}
    for word in words:
        if len(word) >= 2:  # 只考虑2个字符以上的词
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序，取前几个
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, freq in sorted_words[:max_keywords]]
    
    return '，'.join(keywords)


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
    # 修复字符集定义，正确包含中文字符范围
    allowed_chars = set()
    # 添加中文字符范围
    for i in range(0x4e00, 0x9fa6):  # 中文字符范围
        allowed_chars.add(chr(i))
    # 添加英文字母、数字、标点符号
    allowed_chars.update(string.ascii_letters + string.digits + '，。！？；：""''（）【】- ')
    
    text = ''.join(char for char in text if char in allowed_chars)
    
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