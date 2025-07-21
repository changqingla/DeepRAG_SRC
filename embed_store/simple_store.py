#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的向量存储命令行工具

本工具提供简单的命令行界面，用于将已向量化的分块
直接存储到本地Elasticsearch，无需租户系统。

用法:
    python simple_store.py embedded_chunks.json [选项]

作者: RAGFlow开发团队
许可证: Apache 2.0
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from timeit import default_timer as timer

# 导入简化的存储类
try:
    from chunk_store import SimpleVectorStore, SimpleStoreConfig
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在embed_store目录下运行此脚本")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="简化的向量存储工具 - 将向量直接存储到ES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本存储
  python simple_store.py embedded_chunks.json
  
  # 指定索引名称
  python simple_store.py embedded_chunks.json --index my_vectors
  
  # 自定义ES地址
  python simple_store.py embedded_chunks.json --es-host http://192.168.1.100:9200
  
  # 仅验证数据
  python simple_store.py embedded_chunks.json --validate-only
        """
    )
    
    # 输入文件
    parser.add_argument(
        'input_file',
        help='包含已向量化分块的JSON文件'
    )
    
    # 存储配置
    parser.add_argument(
        '--index', '-i',
        default='ragflow_vectors',
        help='ES索引名称 (默认: ragflow_vectors)'
    )
    
    parser.add_argument(
        '--es-host',
        default='http://localhost:9200',
        help='Elasticsearch地址 (默认: http://localhost:9200)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='批量大小 (默认: 8)'
    )
    
    # 操作选项
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证数据，不存储'
    )
    
    parser.add_argument(
        '--show-info',
        action='store_true',
        help='显示索引信息'
    )
    
    parser.add_argument(
        '--delete-index',
        action='store_true',
        help='删除索引 (危险操作!)'
    )
    
    # 其他选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def load_chunks(file_path: Path):
    """加载向量分块数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list):
            raise ValueError("输入文件必须包含分块列表")
        
        logging.info(f"从 {file_path} 加载了 {len(chunks)} 个分块")
        return chunks
        
    except Exception as e:
        logging.error(f"加载文件失败 {file_path}: {e}")
        sys.exit(1)


def validate_chunks(chunks):
    """验证分块数据"""
    print(f"\n数据验证结果:")
    print(f"总分块数: {len(chunks)}")
    
    if not chunks:
        print("❌ 没有分块数据")
        return False
    
    # 检查向量字段
    vector_fields = []
    content_count = 0
    
    for i, chunk in enumerate(chunks):
        # 检查内容
        if chunk.get("content_with_weight"):
            content_count += 1
        
        # 检查向量字段
        for key in chunk.keys():
            if key.startswith("q_") and key.endswith("_vec"):
                if key not in vector_fields:
                    vector_fields.append(key)
                
                vector = chunk[key]
                if not isinstance(vector, list) or len(vector) == 0:
                    print(f"❌ 分块 {i}: 向量字段 {key} 格式错误")
                    return False
    
    print(f"有内容的分块: {content_count}")
    print(f"向量字段: {vector_fields}")
    
    if not vector_fields:
        print("❌ 未找到向量字段")
        return False
    
    print("✅ 数据验证通过")
    return True


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        logging.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    
    # 加载数据
    chunks = load_chunks(input_path)
    
    # 验证数据
    if not validate_chunks(chunks):
        sys.exit(1)
    
    if args.validate_only:
        print("✅ 仅验证模式，验证完成")
        return
    
    # 创建存储配置
    es_config = {
        "hosts": args.es_host,
        "timeout": 600
    }
    
    config = SimpleStoreConfig(
        index_name=args.index,
        es_config=es_config,
        batch_size=args.batch_size
    )
    
    print(f"\n存储配置:")
    print(f"索引名称: {config.index_name}")
    print(f"ES地址: {config.es_config['hosts']}")
    print(f"批量大小: {config.batch_size}")
    
    # 创建存储器
    try:
        store = SimpleVectorStore(config)
        print("✅ ES连接成功")
    except Exception as e:
        print(f"❌ ES连接失败: {e}")
        print("请确保Elasticsearch正在运行")
        sys.exit(1)
    
    # 显示索引信息
    if args.show_info:
        index_info = store.get_index_info()
        print(f"\n索引信息:")
        for key, value in index_info.items():
            print(f"  {key}: {value}")
    
    # 删除索引（如果请求）
    if args.delete_index:
        confirm = input(f"确定要删除索引 '{args.index}' 吗? (yes/no): ")
        if confirm.lower() == 'yes':
            if store.delete_index():
                print("✅ 索引已删除")
            else:
                print("❌ 索引删除失败")
        else:
            print("取消删除操作")
        return
    
    # 存储向量
    print(f"\n开始存储 {len(chunks)} 个向量分块...")
    start_time = timer()
    
    try:
        stored_count, errors = store.store_vectors(chunks)
        processing_time = timer() - start_time
        
        print(f"\n存储结果:")
        print(f"成功存储: {stored_count}/{len(chunks)} 个分块")
        print(f"处理时间: {processing_time:.2f}秒")
        print(f"存储速度: {stored_count/processing_time:.1f} 分块/秒")
        
        if errors:
            print(f"错误数量: {len(errors)}")
            print("错误详情:")
            for error in errors[:5]:  # 只显示前5个错误
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... 还有 {len(errors)-5} 个错误")
        else:
            print("✅ 所有分块存储成功!")
        
        # 显示最终索引信息
        index_info = store.get_index_info()
        print(f"\n最终索引信息:")
        print(f"索引名称: {index_info['index_name']}")
        print(f"向量维度: {index_info['vector_size']}")
        print(f"索引存在: {index_info['index_exists']}")
        
    except Exception as e:
        print(f"❌ 存储失败: {e}")
        logging.exception("存储过程中出现错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
