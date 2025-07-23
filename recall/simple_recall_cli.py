#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的召回CLI工具 - 完全移除ID依赖
保持检索效果完全不变，但移除所有租户ID、知识库ID的复杂性
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入简化的检索器
from simple_retriever import SimpleRetriever, SimpleRetrievalConfig

# 导入向量化和重排序模型
from rag.llm import EmbeddingModel, RerankModel


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_embedding_model(factory: str, model_name: str, base_url: str):
    """创建向量化模型"""
    print(f"创建向量化模型: {factory}/{model_name}")
    print(f"服务地址: {base_url}")
    
    try:
        model = EmbeddingModel[factory](
            key="dummy_key",  # 对于本地服务，key通常不重要
            model_name=model_name,
            base_url=base_url
        )
        print("✅ 向量化模型创建成功")
        return model
    except Exception as e:
        print(f"❌ 向量化模型创建失败: {e}")
        return None


def create_rerank_model(factory: str, model_name: str, base_url: str):
    """创建重排序模型"""
    if not factory or not model_name:
        return None
        
    print(f"创建重排序模型: {factory}")
    print(f"模型名称: {model_name}")
    print(f"服务地址: {base_url}")
    
    try:
        model = RerankModel[factory](
            key="dummy_key",  # 对于本地服务，key通常不重要
            model_name=model_name,
            base_url=base_url
        )
        print("✅ 重排序模型创建成功")
        return model
    except Exception as e:
        print(f"❌ 重排序模型创建失败: {e}")
        return None


def format_results(result: dict, show_vectors: bool = False):
    """格式化显示结果"""
    total = result.get("total", 0)
    chunks = result.get("chunks", [])
    
    print(f"召回完成，总共找到 {total} 个相关分块，返回前 {len(chunks)} 个。")
    print()
    
    if not chunks:
        print("❌ 没有找到相关分块")
        return
    
    print(f"找到 {len(chunks)} 个相关分块:")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"=== 分块 {i} ===")
        print(f"文档: {chunk.get('docnm_kwd', 'Unknown')}")
        
        # 显示页码信息
        page_nums = chunk.get('page_num_int', [])
        if page_nums:
            if isinstance(page_nums, list):
                page_str = f"[{', '.join(map(str, page_nums))}]"
            else:
                page_str = f"[{page_nums}]"
            print(f"页码: {page_str}")
        
        # 显示内容（高亮处理）
        content = chunk.get('content_with_weight', '') or chunk.get('content_ltks', '')
        if content:
            # 简单的内容截断
            if len(content) > 500:
                content = content[:500] + "..."
            print(f"内容(高亮): {content}")
        
        # 显示相似度信息
        similarity = chunk.get('similarity', 0.0)
        print(f"相似度: {similarity:.4f}")
        
        # 显示详细相似度（如果可用）
        if 'term_similarity' in chunk or 'vector_similarity' in chunk:
            term_sim = chunk.get('term_similarity', 0.0)
            vector_sim = chunk.get('vector_similarity', 0.0)
            print(f"文本相似度: {term_sim:.4f}, 向量相似度: {vector_sim:.4f}")
        
        # 显示向量信息（如果需要）
        if show_vectors:
            vector = chunk.get('q_1024_vec', [])
            if vector:
                print(f"向量维度: {len(vector)}")
        
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化的DeepRAG召回测试工具")
    
    # 基本参数
    parser.add_argument("question", nargs="?", help="查询问题")
    parser.add_argument("--index", default="test_documents", help="ES索引名称")
    parser.add_argument("--es-host", default="http://10.0.100.36:9201", help="ES服务地址")
    
    # 召回参数
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    parser.add_argument("--page", type=int, default=1, help="页码")
    parser.add_argument("--similarity", type=float, default=0.2, help="相似度阈值")
    parser.add_argument("--vector-weight", type=float, default=0.3, help="向量相似度权重")
    
    # 向量化模型参数
    parser.add_argument("--model-factory", default="VLLM", help="向量化模型工厂")
    parser.add_argument("--model-name", default="bge-m3", help="向量化模型名称")
    parser.add_argument("--model-base-url", default="http://10.0.1.4:8002/v1", help="向量化模型服务地址")
    
    # 重排序模型参数
    parser.add_argument("--rerank-factory", help="重排序模型工厂")
    parser.add_argument("--rerank-model-name", help="重排序模型名称")
    parser.add_argument("--rerank-base-url", help="重排序模型服务地址")
    
    # 其他参数
    parser.add_argument("--no-highlight", action="store_true", help="禁用高亮")
    parser.add_argument("--show-vectors", action="store_true", help="显示向量信息")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    parser.add_argument("--health", action="store_true", help="健康检查")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 健康检查
    if args.health:
        try:
            config = SimpleRetrievalConfig(
                index_name=args.index,
                es_host=args.es_host
            )
            retriever = SimpleRetriever(config)
            health = retriever.health_check()
            
            print("=== 健康检查结果 ===")
            print(f"状态: {health['status']}")
            print(f"ES状态: {health['elasticsearch']['status']}")
            print(f"索引存在: {health['index']['exists']}")
            print(f"索引名称: {health['index']['name']}")
            
            if health['status'] == 'healthy':
                print("✅ 系统健康")
                sys.exit(0)
            else:
                print("❌ 系统异常")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            sys.exit(1)
    
    # 检查查询问题
    if not args.question:
        print("❌ 请提供查询问题")
        sys.exit(1)
    
    # 显示参数
    print(f"查询问题: {args.question}")
    print(f"搜索索引: {args.index}")
    print(f"召回参数: top_k={args.top_k}, similarity={args.similarity}, vector_weight={args.vector_weight}")
    print()
    
    try:
        # 创建向量化模型
        embedding_model = create_embedding_model(
            args.model_factory, 
            args.model_name, 
            args.model_base_url
        )
        if not embedding_model:
            sys.exit(1)
        
        # 创建重排序模型（可选）
        rerank_model = None
        if args.rerank_factory and args.rerank_model_name:
            rerank_model = create_rerank_model(
                args.rerank_factory,
                args.rerank_model_name,
                args.rerank_base_url
            )
            if rerank_model:
                print("✅ 将使用重排序模型进行结果优化")
            else:
                print("ℹ️  重排序模型创建失败，将使用默认重排序算法")
        else:
            print("ℹ️  未指定重排序模型，将使用默认重排序算法")
        
        # 创建简化检索器
        config = SimpleRetrievalConfig(
            index_name=args.index,
            es_host=args.es_host,
            page=args.page,
            page_size=args.top_k,
            similarity_threshold=args.similarity,
            vector_similarity_weight=args.vector_weight,
            highlight=not args.no_highlight
        )
        
        retriever = SimpleRetriever(config)
        
        # 执行召回
        print("正在执行简化召回...")
        result = retriever.retrieval(
            question=args.question,
            emb_mdl=embedding_model,
            page=args.page,
            page_size=args.top_k,
            similarity_threshold=args.similarity,
            vector_similarity_weight=args.vector_weight,
            rerank_mdl=rerank_model,
            highlight=not args.no_highlight
        )
        
        # 显示结果
        format_results(result, args.show_vectors)
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
