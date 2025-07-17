#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLTK 数据下载脚本

这个脚本会下载 DeepRAG 项目所需的所有 NLTK 数据包。
"""

import nltk
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_data():
    """下载所需的 NLTK 数据包"""
    
    # 需要下载的数据包列表
    required_packages = [
        'punkt',           # 句子分词器
        'punkt_tab',       # 新版本的句子分词器
        'wordnet',         # WordNet 词汇数据库
        'averaged_perceptron_tagger',  # 词性标注器
        'stopwords',       # 停用词
        'omw-1.4',         # 开放多语言词网
    ]
    
    logger.info("开始下载 NLTK 数据包...")
    
    for package in required_packages:
        try:
            logger.info(f"正在下载 {package}...")
            nltk.download(package, quiet=False)
            logger.info(f"✓ {package} 下载成功")
        except Exception as e:
            logger.error(f"✗ {package} 下载失败: {e}")
    
    logger.info("NLTK 数据包下载完成！")

def verify_nltk_data():
    """验证 NLTK 数据是否正确安装"""
    
    logger.info("验证 NLTK 数据安装...")
    
    try:
        # 测试句子分词
        from nltk.tokenize import sent_tokenize, word_tokenize
        test_text = "Hello world. This is a test sentence."
        sentences = sent_tokenize(test_text)
        words = word_tokenize(test_text)
        logger.info(f"✓ 句子分词测试成功: {len(sentences)} 个句子")
        logger.info(f"✓ 词语分词测试成功: {len(words)} 个词语")
        
        # 测试词形还原
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        test_word = lemmatizer.lemmatize("running", pos='v')
        logger.info(f"✓ 词形还原测试成功: running -> {test_word}")
        
        logger.info("✓ 所有 NLTK 功能验证成功！")
        return True
        
    except Exception as e:
        logger.error(f"✗ NLTK 功能验证失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("DeepRAG NLTK 数据下载工具")
    print("=" * 50)
    
    # 下载数据
    download_nltk_data()
    
    print("\n" + "=" * 50)
    
    # 验证安装
    if verify_nltk_data():
        print("🎉 NLTK 设置完成！现在可以运行 DeepRAG 了。")
    else:
        print("❌ NLTK 设置失败，请检查网络连接或手动下载。")
    
    print("=" * 50)
