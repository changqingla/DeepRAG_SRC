#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同义词模块测试脚本

独立的测试脚本，用于测试同义词功能，无需复杂的路径设置。

使用方法:
python test_synonym.py
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_synonym_module():
    """测试同义词模块功能"""
    print("🔍 测试同义词模块")
    print("=" * 50)
    
    try:
        # 导入同义词模块
        from rag.nlp.synonym import Dealer
        print("✅ 成功导入同义词模块")
        
        # 创建同义词处理器（不使用Redis）
        print("\n📚 创建同义词处理器...")
        dealer = Dealer()
        print("✅ 同义词处理器创建成功")
        
        # 检查词典加载情况
        print(f"\n📊 词典状态:")
        print(f"词典大小: {len(dealer.dictionary)} 个词汇")
        
        if dealer.dictionary:
            print("✅ 词典加载成功")
            # 显示前5个词汇示例
            sample_words = list(dealer.dictionary.keys())[:5]
            print(f"词典示例: {sample_words}")
        else:
            print("⚠️ 词典为空或未加载")
        
        # 测试英文同义词查找
        print(f"\n🔤 测试英文同义词查找:")
        english_words = ["good", "bad", "big", "small"]
        
        for word in english_words:
            try:
                synonyms = dealer.lookup(word, topn=5)
                print(f"'{word}' 的同义词: {synonyms}")
            except Exception as e:
                print(f"'{word}' 查找失败: {e}")
        
        # 测试中文同义词查找
        print(f"\n🀄 测试中文同义词查找:")
        
        if dealer.dictionary:
            # 使用词典中的词汇进行测试
            test_words = list(dealer.dictionary.keys())[:3]
            for word in test_words:
                try:
                    synonyms = dealer.lookup(word, topn=5)
                    print(f"'{word}' 的同义词: {synonyms}")
                except Exception as e:
                    print(f"'{word}' 查找失败: {e}")
        else:
            # 使用常见中文词汇测试
            chinese_words = ["人工智能", "机器学习", "深度学习"]
            for word in chinese_words:
                try:
                    synonyms = dealer.lookup(word, topn=5)
                    print(f"'{word}' 的同义词: {synonyms}")
                except Exception as e:
                    print(f"'{word}' 查找失败: {e}")
        
        # 测试不同参数
        print(f"\n⚙️ 测试不同参数:")
        test_word = "good"
        
        # 测试不同的topn值
        for topn in [3, 5, 10]:
            synonyms = dealer.lookup(test_word, topn=topn)
            print(f"'{test_word}' (top{topn}): {synonyms}")
        
        print(f"\n🎉 同义词模块测试完成!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保在DeepRAG_SRC目录下运行此脚本")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_wordnet():
    """测试NLTK WordNet功能"""
    print(f"\n📖 测试NLTK WordNet:")
    
    try:
        from nltk.corpus import wordnet
        
        # 测试WordNet是否可用
        synsets = wordnet.synsets("good")
        if synsets:
            print("✅ NLTK WordNet 可用")
            print(f"'good' 的同义词集数量: {len(synsets)}")
            
            # 提取同义词
            synonyms = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().replace('_', ' '))
            
            unique_synonyms = list(set(synonyms))
            print(f"'good' 的同义词: {unique_synonyms[:10]}")  # 显示前10个
        else:
            print("⚠️ WordNet 没有找到 'good' 的同义词")
            
    except ImportError:
        print("❌ NLTK WordNet 不可用，请安装: pip install nltk")
    except Exception as e:
        print(f"❌ WordNet 测试失败: {e}")

def main():
    """主函数"""
    print("🚀 DeepRAG 同义词模块测试")
    print("=" * 60)
    
    # 显示当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本位置: {Path(__file__).parent.absolute()}")
    
    # 测试同义词模块
    success = test_synonym_module()
    
    # 测试WordNet
    test_wordnet()
    
    if success:
        print(f"\n✅ 所有测试完成!")
    else:
        print(f"\n❌ 测试失败，请检查环境配置")

if __name__ == "__main__":
    main()
