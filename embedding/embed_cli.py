# -*- coding: utf-8 -*-
"""
本脚本为使用嵌入模型进行文档分块嵌入提供命令行界面，

使用方法:
    python embed_cli.py chunks.json [选项]

功能特性:
- 支持多种嵌入模型（BAAI、OpenAI、通义千问等）
- 批量处理文档分块
- 嵌入结果分析和相似度计算
- 多格式数据导出

作者: HU TAO
许可证: Apache 2.0
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict, Any

# 导入本地模块
from chunk_embedder import ChunkEmbedder, EmbeddingConfig
from embedding_utils import (
    EmbeddingConfigManager, EmbeddingAnalyzer, EmbeddingExporter,
    EmbeddingResult
)


def setup_logging(verbose: bool = False):
    """
    设置日志配置

    Args:
        verbose (bool): 是否启用详细日志输出
    """
    # 根据详细程度设置日志级别
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(
        description="DeepRAG 分块嵌入器 - 为文档分块生成嵌入向量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认模型嵌入分块
  python embed_cli.py chunks.json --output embedded_chunks.json

  # 使用 OpenAI 嵌入模型
  python embed_cli.py chunks.json --config openai_small --api-key YOUR_KEY

  # 使用自定义模型配置
  python embed_cli.py chunks.json --model-factory BAAI --model-name BAAI/bge-small-en-v1.5

  # 分析嵌入结果
  python embed_cli.py chunks.json --analyze --output analysis.json

  # 仅导出向量
  python embed_cli.py chunks.json --export-vectors vectors.npy
        """
    )
    
    # 输入/输出选项
    parser.add_argument(
        'input_file',
        nargs='?',
        help='包含文档分块的输入 JSON 文件'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='输出文件路径（默认：自动生成）'
    )
    
    # 模型配置选项
    parser.add_argument(
        '--config',
        help='使用预定义配置（简化版本中未实现）'
    )
    
    parser.add_argument(
        '--model-factory',
        help='模型工厂名称（如 BAAI、OpenAI、Tongyi-Qianwen）'
    )
    
    parser.add_argument(
        '--model-name',
        help='模型名称'
    )
    
    parser.add_argument(
        '--api-key',
        help='模型的 API 密钥（如果需要）'
    )
    
    parser.add_argument(
        '--base-url',
        help='模型 API 的基础 URL（如果需要）'
    )

    parser.add_argument(
        '--model-base-url',
        help='模型 API 的基础 URL（--base-url 的别名）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='处理批次大小（默认：16）'
    )
    
    parser.add_argument(
        '--filename-weight',
        type=float,
        default=0.1,
        help='文件名嵌入的权重（默认：0.1）'
    )
    
    # 分析选项
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='分析嵌入结果'
    )
    
    parser.add_argument(
        '--find-similar',
        type=int,
        help='查找与第一个分块相似的分块（指定返回的前 k 个）'
    )
    
    parser.add_argument(
        '--cluster',
        type=int,
        help='对分块进行聚类（指定聚类数量）'
    )
    
    # 导出选项
    parser.add_argument(
        '--export-vectors',
        help='仅导出向量到指定文件'
    )
    
    parser.add_argument(
        '--export-format',
        choices=['npy', 'txt', 'csv'],
        default='npy',
        help='向量导出格式（默认：npy）'
    )
    
    parser.add_argument(
        '--export-metadata',
        help='导出带有元数据的分块到指定文件'
    )
    
    # 配置管理选项
    parser.add_argument(
        '--save-config',
        help='将当前配置保存到文件'
    )
    
    parser.add_argument(
        '--load-config',
        help='从文件加载配置'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='列出可用的预定义配置'
    )
    
    # 其他选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    return parser.parse_args()


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载分块数据

    Args:
        file_path (Path): JSON 文件路径

    Returns:
        List[Dict[str, Any]]: 分块数据列表
    """
    try:
        # 以 UTF-8 编码读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # 验证数据格式
        if not isinstance(chunks, list):
            raise ValueError("输入文件必须包含分块列表")

        logging.info(f"从 {file_path} 加载了 {len(chunks)} 个分块")
        return chunks

    except Exception as e:
        logging.error(f"从 {file_path} 加载分块失败: {e}")
        sys.exit(1)


def create_embedding_config(args) -> EmbeddingConfig:
    """
    根据命令行参数创建嵌入配置

    Args:
        args: 命令行参数对象

    Returns:
        EmbeddingConfig: 嵌入配置实例
    """
    # 处理 base_url 别名（支持两种参数名）
    base_url = args.base_url or args.model_base_url or ""

    if args.model_factory and args.model_name:
        # 使用自定义配置创建嵌入配置
        return EmbeddingConfigManager.create_custom_config(
            model_factory=args.model_factory,
            model_name=args.model_name,
            api_key=args.api_key or "",
            base_url=base_url,
            batch_size=args.batch_size,
            filename_embd_weight=args.filename_weight
        )
    else:
        # 必须指定模型参数，否则抛出错误
        raise ValueError(
            "必须指定嵌入模型参数！\n\n"
            "必需参数：\n"
            "  --model-factory: 模型工厂名称（如 VLLM, OpenAI, LocalAI 等）\n"
            "  --model-name: 具体的模型名称\n\n"
            "可选参数：\n"
            "  --api-key: API 密钥（某些服务需要）\n"
            "  --model-base-url: 服务端点 URL（本地服务需要）\n\n"
            "示例：\n"
            "  # VLLM 服务\n"
            "  python embed_cli.py chunks.json --model-factory VLLM --model-name bge-m3 --model-base-url http://localhost:8002/v1\n\n"
            "  # OpenAI 服务\n"
            "  python embed_cli.py chunks.json --model-factory OpenAI --model-name text-embedding-3-small --api-key YOUR_KEY\n\n"
            "  # LocalAI 服务\n"
            "  python embed_cli.py chunks.json --model-factory LocalAI --model-name bge-m3 --model-base-url http://localhost:8080/v1"
        )


def get_output_path(input_path: Path, output_arg: str, suffix: str = "_embedded") -> Path:
    """
    生成输出文件路径

    Args:
        input_path (Path): 输入文件路径
        output_arg (str): 用户指定的输出路径
        suffix (str): 默认后缀名

    Returns:
        Path: 输出文件路径
    """
    # 如果用户指定了输出路径，直接使用
    if output_arg:
        return Path(output_arg)

    # 否则自动生成输出路径（在输入文件名后添加后缀）
    return input_path.parent / f"{input_path.stem}{suffix}.json"


def main():
    """
    主函数 - 命令行工具的入口点

    处理命令行参数，执行嵌入操作，并输出结果。
    """
    # 解析命令行参数
    args = parse_arguments()

    # 设置日志配置
    setup_logging(args.verbose)

    # 如果请求列出配置，则显示信息并退出
    if args.list_configs:
        print("\n简化版本不支持预定义配置")
        print("请使用 --model-factory 和 --model-name 参数指定模型")
        return

    # 检查是否提供了输入文件（除了 --list-configs 之外都需要）
    if not args.input_file:
        logging.error("需要提供输入文件")
        sys.exit(1)

    # 加载输入分块数据
    input_path = Path(args.input_file)
    if not input_path.exists():
        logging.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    
    chunks = load_chunks(input_path)
    if not chunks:
        logging.error("输入文件中没有找到分块数据")
        sys.exit(1)
    
    # 创建嵌入配置
    try:
        config = create_embedding_config(args)
        logging.info(f"使用嵌入模型: {config.model_factory}/{config.model_name}")
    except Exception as e:
        logging.error(f"创建嵌入配置失败: {e}")
        sys.exit(1)
    
    # 如果请求保存配置（简化版本中未实现）
    if args.save_config:
        logging.warning("简化版本中未实现配置保存功能")
    
    # 初始化嵌入器
    try:
        embedder = ChunkEmbedder(config)
    except Exception as e:
        logging.error(f"初始化嵌入器失败: {e}")
        sys.exit(1)
    
    # 执行嵌入操作
    logging.info("开始嵌入处理...")
    start_time = timer()
    
    try:
        # 同步执行嵌入操作
        token_count, vector_size = embedder.embed_chunks_sync(chunks)
        processing_time = timer() - start_time
        
        logging.info(f"嵌入处理完成，耗时 {processing_time:.2f}s")
        logging.info(f"处理了 {token_count} 个 token，向量维度: {vector_size}")
        
    except Exception as e:
        logging.error(f"嵌入分块失败: {e}")
        sys.exit(1)
    
    # 创建结果对象
    result = EmbeddingResult(
        chunks=chunks,
        token_count=token_count,
        vector_size=vector_size,
        processing_time=processing_time,
        model_info={
            "factory": config.model_factory,
            "name": config.model_name,
            "vector_size": vector_size
        }
    )
    
    # 保存嵌入后的分块数据
    output_path = get_output_path(input_path, args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logging.info(f"嵌入后的分块已保存到 {output_path}")
    
    # 执行分析（如果请求）
    if args.analyze:
        analysis = EmbeddingAnalyzer.analyze_embeddings(chunks)
        analysis_path = get_output_path(input_path, None, "_analysis")
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n嵌入分析结果:")
        print(f"总分块数: {analysis.get('total_chunks', 0)}")
        print(f"向量维度: {analysis.get('vector_dimension', 0)}")
        print(f"分析结果已保存到: {analysis_path}")
    
    # 查找相似分块（如果请求）
    if args.find_similar and len(chunks) > 1:
        similar = EmbeddingAnalyzer.find_similar_chunks(
            chunks[1:], chunks[0], top_k=args.find_similar
        )
        
        print(f"\n与第一个分块最相似的前 {args.find_similar} 个分块:")
        for i, (idx, similarity, chunk) in enumerate(similar, 1):
            content_preview = chunk.get("content_with_weight", "")[:100]
            print(f"{i}. 相似度: {similarity:.4f} - {content_preview}...")
    
    # 执行聚类（如果请求）
    if args.cluster:
        cluster_result = EmbeddingAnalyzer.cluster_chunks(chunks, n_clusters=args.cluster)
        cluster_path = get_output_path(input_path, None, "_clusters")
        
        with open(cluster_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n聚类结果:")
        print(f"轮廓系数: {cluster_result.get('silhouette_score', 0):.4f}")
        print(f"聚类大小: {cluster_result.get('cluster_sizes', {})}")
        print(f"结果已保存到: {cluster_path}")
    
    # 导出向量（如果请求）
    if args.export_vectors:
        EmbeddingExporter.export_vectors_only(
            chunks, Path(args.export_vectors), format=args.export_format
        )
    
    # 导出元数据（如果请求）
    if args.export_metadata:
        EmbeddingExporter.export_with_metadata(chunks, Path(args.export_metadata))
    
    # 打印处理摘要
    print(f"\n{'='*60}")
    print(f"嵌入处理摘要")
    print(f"{'='*60}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"使用模型: {config.model_factory}/{config.model_name}")
    print(f"处理分块数: {len(chunks)}")
    print(f"使用 token 数: {token_count:,}")
    print(f"向量维度: {vector_size}")
    print(f"处理时间: {processing_time:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
