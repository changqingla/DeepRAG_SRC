#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
召回命令行工具，提供命令行界面来测试召回功能

作者: Hu Tao
许可证: Apache 2.0
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List

# 添加DeepRag根目录到路径
current_dir = Path(__file__).parent.absolute()
DeepRag_root = current_dir.parent
sys.path.insert(0, str(DeepRag_root))

# 导入召回器
from retriever import DeepRagPureRetriever, DeepRagRetrievalConfig

# 导入向量化模型
from rag.llm import EmbeddingModel




def create_embedding_model_from_args(args):
    """根据命令行参数创建向量化模型"""
    print(f"创建向量化模型: {args.model_factory}/{args.model_name}")
    print(f"服务地址: {args.model_base_url}")

    # 检查模型工厂是否可用
    if args.model_factory not in EmbeddingModel:
        available_factories = list(EmbeddingModel.keys())
        raise ValueError(f"不支持的嵌入模型工厂: {args.model_factory}. 可用工厂: {available_factories}")

    model_class = EmbeddingModel[args.model_factory]

    # 根据模型类型准备初始化参数
    if args.model_factory in ["LocalAI", "VLLM", "OpenAI-API-Compatible", "LM-Studio", "GPUStack"]:
        # 这些模型需要特定的参数顺序: key, model_name, base_url
        if not args.model_base_url:
            raise ValueError(f"{args.model_factory} 嵌入模型需要 base_url 参数")

        model = model_class(
            getattr(args, 'api_key', '') or "empty",  # key 作为位置参数
            args.model_name,                          # model_name 作为位置参数
            args.model_base_url                       # base_url 作为位置参数
        )
    else:
        # 其他模型的标准初始化
        init_params = {
            "model_name": args.model_name,
        }

        # 添加可选参数
        if hasattr(args, 'api_key') and args.api_key:
            init_params["key"] = args.api_key
        if hasattr(args, 'model_base_url') and args.model_base_url:
            init_params["base_url"] = args.model_base_url

        # 初始化模型
        model = model_class(**init_params)

    print("✅ 向量化模型创建成功")
    return model


def create_rerank_model_from_args(args):
    """根据命令行参数创建重排序模型"""
    if not args.rerank_factory:
        return None

    try:
        print(f"创建重排序模型: {args.rerank_factory}")
        if args.rerank_model_name:
            print(f"模型名称: {args.rerank_model_name}")
        if args.rerank_base_url:
            print(f"服务地址: {args.rerank_base_url}")

        # 导入rerank模型字典
        from rag.llm import RerankModel

        # 检查重排序模型工厂是否可用
        if args.rerank_factory not in RerankModel:
            available_factories = list(RerankModel.keys())
            raise ValueError(f"不支持的重排序模型工厂: {args.rerank_factory}. 可用工厂: {available_factories}")

        # 获取对应的rerank类
        rerank_class = RerankModel[args.rerank_factory]

        # 准备参数
        key = getattr(args, 'rerank_api_key', None) or "empty"  # 使用默认值避免空字符串
        model_name = args.rerank_model_name or ""
        base_url = args.rerank_base_url

        # 根据模型类型准备初始化参数
        if args.rerank_factory in ["LocalAI", "VLLM", "OpenAI-API-Compatible", "LM-Studio", "GPUStack"]:
            # 这些模型需要特定的参数顺序: key, model_name, base_url
            if not base_url:
                raise ValueError(f"{args.rerank_factory} 重排序模型需要 base_url 参数")

            model = rerank_class(
                key,                # key 作为位置参数
                model_name,         # model_name 作为位置参数
                base_url           # base_url 作为位置参数
            )
        else:
            # 其他模型的标准初始化
            init_params = {
                "key": key,
                "model_name": model_name,
            }

            # 添加可选参数
            if base_url:
                init_params["base_url"] = base_url

            # 初始化模型
            model = rerank_class(**init_params)

        print(f"✅ 重排序模型创建成功")
        return model

    except Exception as e:
        print(f"❌ 重排序模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepRag召回工具 - 基于DeepRag原有算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本召回
  python recall_cli.py "什么是人工智能？"

  # 指定向量化模型
  python recall_cli.py "机器学习算法" \\
    --model-factory VLLM \\
    --model-name bge-m3 \\
    --model-base-url http://10.0.1.4:8002/v1

  # 指定索引和参数
  python recall_cli.py "深度学习" \\
    --indices DeepRag_vectors my_docs \\
    --top-k 20 --similarity 0.3

  # 输出到文件
  python recall_cli.py "神经网络" --output result.json --format json
        """
    )
    
    # 查询参数
    parser.add_argument(
        'question',
        nargs='?',  # 使question参数可选
        help='查询问题'
    )
    
    # 索引配置
    parser.add_argument(
        '--indices', '-i',
        nargs='+',
        default=['test_documents'],
        help='ES索引名称列表 (默认: test_documents)'
    )
    
    # 召回参数
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='返回top-k结果 (默认: 10)'
    )
    
    parser.add_argument(
        '--similarity', '-s',
        type=float,
        default=0.2,
        help='相似度阈值 (默认: 0.2)'
    )
    
    parser.add_argument(
        '--vector-weight', '-w',
        type=float,
        default=0.3,
        help='向量相似度权重 (默认: 0.3)'
    )
    
    parser.add_argument(
        '--page',
        type=int,
        default=1,
        help='页码 (默认: 1)'
    )
    
    parser.add_argument(
        '--doc-ids',
        nargs='+',
        help='指定文档ID列表'
    )
    
    # 输出选项
    parser.add_argument(
        '--output', '-o',
        help='输出文件路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'text', 'simple'],
        default='text',
        help='输出格式 (默认: text)'
    )
    
    parser.add_argument(
        '--no-highlight',
        action='store_true',
        help='禁用高亮显示'
    )
    
    # 向量化模型参数
    parser.add_argument(
        '--model-factory',
        default='VLLM',
        help='向量化模型工厂 (默认: VLLM)'
    )

    parser.add_argument(
        '--model-name',
        default='bge-m3',
        help='向量化模型名称 (默认: bge-m3)'
    )

    parser.add_argument(
        '--model-base-url',
        default='http://10.0.1.4:8002/v1',
        help='向量化模型服务地址 (默认: http://10.0.1.4:8002/v1)'
    )

    # 重排序模型参数
    parser.add_argument(
        '--rerank-factory',
        help='重排序模型工厂类型'
    )

    parser.add_argument(
        '--rerank-model-name',
        help='重排序模型名称'
    )

    parser.add_argument(
        '--rerank-base-url',
        help='重排序模型服务地址'
    )

    parser.add_argument(
        '--rerank-api-key',
        help='重排序模型API密钥'
    )

    # 其他选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--health',
        action='store_true',
        help='检查健康状态'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='干运行模式（不连接ES，仅测试配置）'
    )
    
    return parser.parse_args()


def format_chunks_text(chunks: List[dict], highlight: bool = True) -> str:
    """格式化分块为文本输出"""
    if not chunks:
        return "没有找到相关分块。"
    
    output = []
    output.append(f"找到 {len(chunks)} 个相关分块:\n")
    
    for i, chunk in enumerate(chunks, 1):
        output.append(f"=== 分块 {i} ===")
        output.append(f"文档: {chunk.get('docnm_kwd', 'Unknown')}")
        output.append(f"页码: {chunk.get('page_num_int', [])}")
        
        # 显示内容
        if highlight and 'highlight' in chunk:
            content = chunk['highlight']
            output.append(f"内容(高亮): {content}")
        else:
            content = chunk.get('content_with_weight', '')
            output.append(f"内容: {content}")
        
        # 显示分数
        if 'similarity' in chunk:
            output.append(f"相似度: {chunk['similarity']:.4f}")
        
        output.append("")  # 空行分隔
    
    return "\n".join(output)


def format_chunks_simple(chunks: List[dict]) -> str:
    """简单格式输出"""
    if not chunks:
        return "没有找到相关分块。"
    
    output = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get('content_with_weight', '')[:100]
        doc_name = chunk.get('docnm_kwd', 'Unknown')
        similarity = chunk.get('similarity', 0)
        output.append(f"{i}. [{doc_name}] {content}... (相似度: {similarity:.3f})")
    
    return "\n".join(output)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 健康检查
    if args.health:
        try:
            config = DeepRagRetrievalConfig(
                index_names=args.indices,
                es_config={
                    "hosts": "http://10.0.100.36:9201",
                    "timeout": 600
                }
            )
            retriever = DeepRagPureRetriever(config)
            health_info = retriever.health_check()
            
            print("=== 健康检查 ===")
            # 安全地打印健康信息
            try:
                print(json.dumps(health_info, indent=2, ensure_ascii=False, default=str))
            except Exception:
                # 如果JSON序列化失败，使用简单格式
                print(f"状态: {health_info.get('status', 'unknown')}")
                if 'components' in health_info:
                    print("组件状态:")
                    for comp, status in health_info['components'].items():
                        print(f"  {comp}: {'✅' if status else '❌'}")
                if 'indices' in health_info:
                    print("索引状态:")
                    for idx, status in health_info['indices'].items():
                        print(f"  {idx}: {'✅' if status else '❌'}")
            return
            
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            sys.exit(1)

    # 检查是否提供了问题参数
    if not args.question:
        print("❌ 请提供查询问题")
        sys.exit(1)

    # 创建召回器配置
    config = DeepRagRetrievalConfig(
        index_names=args.indices,
        page=args.page,
        page_size=args.top_k,
        similarity_threshold=args.similarity,
        vector_similarity_weight=args.vector_weight,
        highlight=not args.no_highlight,
        doc_ids=args.doc_ids,
        es_config={
            "hosts": "http://10.0.100.36:9201",
            "timeout": 600
        }
    )
    
    print(f"查询问题: {args.question}")
    print(f"搜索索引: {args.indices}")
    print(f"召回参数: top_k={args.top_k}, similarity={args.similarity}, vector_weight={args.vector_weight}")
    print()
    
    try:
        # 创建召回器
        retriever = DeepRagPureRetriever(config)

        # 获取向量化模型
        embedding_model = create_embedding_model_from_args(args)

        # 获取重排序模型（可选）
        rerank_model = create_rerank_model_from_args(args)
        if rerank_model:
            print("✅ 将使用重排序模型进行结果优化")
        else:
            print("ℹ️  未指定重排序模型，将使用默认重排序算法")

        # 执行召回
        print("正在执行DeepRag召回...")
        result = retriever.retrieval(
            question=args.question,
            embd_mdl=embedding_model,
            page=args.page,
            page_size=args.top_k,
            similarity_threshold=args.similarity,
            vector_similarity_weight=args.vector_weight,
            top=1024,
            doc_ids=args.doc_ids,
            rerank_mdl=rerank_model,  # 传递重排序模型
            highlight=not args.no_highlight
        )
        
        if "error" in result:
            print(f"❌ 召回失败: {result['error']}")
            sys.exit(1)
        
        chunks = result.get("chunks", [])
        total = result.get("total", 0)
        
        print(f"召回完成，总共找到 {total} 个相关分块，返回前 {len(chunks)} 个。\n")
        
        # 格式化输出
        if args.format == 'json':
            output_text = json.dumps(result, indent=2, ensure_ascii=False)
        elif args.format == 'simple':
            output_text = format_chunks_simple(chunks)
        else:  # text
            output_text = format_chunks_text(chunks, not args.no_highlight)
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.format == 'json':
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    f.write(output_text)
            print(f"结果已保存到: {args.output}")
        else:
            print(output_text)
        
    except Exception as e:
        logging.error(f"召回失败: {e}")
        print(f"❌ 召回失败: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
