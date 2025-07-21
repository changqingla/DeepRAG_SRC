#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的向量存储示例

本示例展示如何使用简化的向量存储器将解析好的向量
直接存储到本地Elasticsearch中，无需租户系统和知识库层级。

作者: RAGFlow开发团队
许可证: Apache 2.0
"""

import json
import logging
from pathlib import Path
from timeit import default_timer as timer

# 导入简化的存储类
from chunk_store import SimpleVectorStore, SimpleStoreConfig


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def example_simple_storage():
    """示例1: 最简单的向量存储"""
    print("=" * 60)
    print("示例1: 最简单的向量存储")
    print("=" * 60)
    
    # 创建简化配置（连接本地ES）
    config = SimpleStoreConfig.create_simple(
        index_name="my_vectors",  # 简单的索引名
        batch_size=4
    )
    
    print(f"索引名称: {config.index_name}")
    print(f"ES地址: {config.es_config['hosts']}")
    
    # 创建简化的向量存储器
    store = SimpleVectorStore(config)
    
    # 模拟已向量化的分块数据
    chunks = [
        {
            "docnm_kwd": "测试文档.pdf",
            "content_with_weight": "这是第一个测试分块的内容。",
            "q_1024_vec": [0.1] * 1024  # 1024维向量
        },
        {
            "docnm_kwd": "测试文档.pdf", 
            "content_with_weight": "这是第二个测试分块的内容。",
            "q_1024_vec": [0.2] * 1024  # 1024维向量
        }
    ]
    
    # 存储向量
    print("开始存储向量...")
    start_time = timer()
    stored_count, errors = store.store_vectors(chunks)
    processing_time = timer() - start_time
    
    print(f"存储结果: {stored_count}/{len(chunks)} 个向量")
    print(f"处理时间: {processing_time:.2f}秒")
    
    if errors:
        print(f"错误: {errors}")
    else:
        print("✅ 存储成功!")
    
    # 显示索引信息
    index_info = store.get_index_info()
    print(f"\n索引信息:")
    for key, value in index_info.items():
        print(f"  {key}: {value}")
    
    return store


def example_load_from_file():
    """示例2: 从文件加载向量数据"""
    print("\n" + "=" * 60)
    print("示例2: 从文件加载向量数据")
    print("=" * 60)
    
    # 检查是否有embedding模块的输出文件
    embedded_file = Path("../embedding/embedded_chunks.json")
    
    if not embedded_file.exists():
        print(f"文件不存在: {embedded_file}")
        print("请先运行embedding模块生成向量数据")
        return None
    
    # 加载向量数据
    with open(embedded_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"从文件加载了 {len(chunks)} 个向量分块")
    
    # 创建配置
    config = SimpleStoreConfig.create_simple(
        index_name="file_vectors",
        batch_size=2
    )
    
    # 存储
    store = SimpleVectorStore(config)
    stored_count, errors = store.store_vectors(chunks)
    
    print(f"存储结果: {stored_count}/{len(chunks)} 个向量")
    
    if not errors:
        print("✅ 文件向量存储成功!")
    
    return store


def example_custom_es_config():
    """示例3: 自定义ES配置"""
    print("\n" + "=" * 60)
    print("示例3: 自定义ES配置")
    print("=" * 60)
    
    # 自定义ES配置
    custom_es_config = {
        "hosts": "http://localhost:9200",  # 可以修改为其他地址
        "timeout": 300,
        # 如果ES有认证，可以添加：
        # "username": "elastic",
        # "password": "password"
    }
    
    config = SimpleStoreConfig(
        index_name="custom_vectors",
        es_config=custom_es_config,
        batch_size=8
    )
    
    print(f"自定义ES配置: {custom_es_config}")
    
    # 测试连接
    try:
        store = SimpleVectorStore(config)
        print("✅ ES连接成功")
        
        # 获取ES健康状态
        health = store.es_conn.health()
        print(f"ES状态: {health.get('status', 'unknown')}")
        
        return store
        
    except Exception as e:
        print(f"❌ ES连接失败: {e}")
        print("请确保Elasticsearch正在运行在指定地址")
        return None


def example_index_management():
    """示例4: 索引管理"""
    print("\n" + "=" * 60)
    print("示例4: 索引管理")
    print("=" * 60)
    
    config = SimpleStoreConfig.create_simple("test_management")
    store = SimpleVectorStore(config)
    
    # 检查索引是否存在
    index_info = store.get_index_info()
    print(f"索引存在: {index_info['index_exists']}")
    
    if index_info['index_exists']:
        print("索引已存在，可以直接存储数据")
    else:
        print("索引不存在，将在首次存储时自动创建")
    
    # 如果需要删除索引（谨慎使用！）
    # store.delete_index()
    
    return store


def main():
    """运行所有示例"""
    setup_logging()
    
    print("RAGFlow简化向量存储示例")
    print("=" * 60)
    print("注意: 请确保Elasticsearch正在运行在 localhost:9200")
    print("如果没有ES，请参考 deploy_elasticsearch.md 部署")
    print()
    
    try:
        # 运行示例
        example_simple_storage()
        example_load_from_file()
        example_custom_es_config()
        example_index_management()
        
        print("\n" + "=" * 60)
        print("所有示例完成!")
        print("=" * 60)
        print("\n使用说明:")
        print("1. 这些示例展示了如何使用简化的向量存储")
        print("2. 不需要租户系统和知识库层级")
        print("3. 直接将向量存储到指定的ES索引")
        print("4. 保持了RAGFlow的核心存储算法")
        
    except Exception as e:
        print(f"\n❌ 示例执行失败: {e}")
        print("\n可能的原因:")
        print("1. Elasticsearch未运行")
        print("2. ES连接配置错误")
        print("3. 缺少必要的依赖")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
