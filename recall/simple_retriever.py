#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的检索器 - 完全移除ID依赖
保持检索效果完全不变，但移除所有租户ID、知识库ID的复杂性
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要的模块
from embed_store.es_connection import SimpleESConnection
from es_adapter import ESAdapter
from rag.nlp import query
from rag.utils.doc_store_conn import MatchDenseExpr, FusionExpr, OrderByExpr

logger = logging.getLogger(__name__)


class SimpleRetrievalConfig:
    """简化的检索配置"""
    
    def __init__(self, 
                 index_name: str = "test_documents",
                 es_host: str = "http://10.0.100.36:9201",
                 page: int = 1,
                 page_size: int = 10,
                 similarity_threshold: float = 0.2,
                 vector_similarity_weight: float = 0.3,
                 top_k: int = 1024,
                 highlight: bool = True,
                 rerank_page_limit: int = 3):
        """
        初始化简化配置
        
        Args:
            index_name: ES索引名称
            es_host: ES服务地址
            page: 页码
            page_size: 每页大小
            similarity_threshold: 相似度阈值
            vector_similarity_weight: 向量相似度权重
            top_k: 向量召回top-k
            highlight: 是否高亮
            rerank_page_limit: 重排序页面限制
        """
        self.index_name = index_name
        self.es_host = es_host
        self.page = page
        self.page_size = page_size
        self.similarity_threshold = similarity_threshold
        self.vector_similarity_weight = vector_similarity_weight
        self.top_k = top_k
        self.highlight = highlight
        self.rerank_page_limit = rerank_page_limit


class SimpleRetriever:
    """简化的检索器 - 无ID依赖"""
    
    def __init__(self, config: SimpleRetrievalConfig):
        """
        初始化简化检索器
        
        Args:
            config: 简化配置对象
        """
        self.config = config
        
        # 创建ES连接
        self.es_conn = SimpleESConnection(config.es_host)
        
        # 创建ES适配器
        self.es_adapter = ESAdapter(self.es_conn)
        
        # 创建查询器
        self.qryr = query.FulltextQueryer()
        
        logger.info(f"简化检索器初始化完成，索引: {config.index_name}")
    
    def get_vector(self, question: str, emb_mdl, topk: int, similarity: float):
        """获取向量表达式"""
        # 对问题进行向量化
        q_vec, _ = emb_mdl.encode([question])
        q_vec = q_vec[0].tolist()
        
        # 构建向量匹配表达式
        return MatchDenseExpr(
            f"q_{len(q_vec)}_vec",  # vector_column_name
            q_vec,                  # embedding_data
            'float',                # embedding_data_type
            'cosine',               # distance_type
            topk,                   # topn
            {"similarity": similarity}  # extra_options
        )
    
    def search(self, question: str, emb_mdl=None, highlight: bool = False,
               page: int = 1, page_size: int = 10, doc_ids: List[str] = None):
        """
        执行搜索（完全复用DeepRag的search逻辑，但移除ID依赖）
        
        Args:
            question: 查询问题
            emb_mdl: 向量化模型
            highlight: 是否高亮
            page: 页码
            page_size: 每页大小
            doc_ids: 指定文档ID列表
            
        Returns:
            搜索结果对象
        """
        if not question:
            # 返回空结果
            class EmptyResult:
                total = 0
                ids = []
                field = {}
                highlight = {}
                aggregation = {}
                keywords = []
                query_vector = []
            return EmptyResult()
        
        # 源字段
        src = [
            "docnm_kwd", "content_ltks", "img_id", "title_tks", 
            "important_kwd", "position_int", "doc_id", "page_num_int", 
            "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
            "question_kwd", "question_tks", "available_int", "content_with_weight"
        ]
        
        # 高亮字段
        highlightFields = ["content_ltks", "title_tks"] if highlight else []
        
        # 过滤条件
        filters = {"available_int": 1}
        if doc_ids:
            filters["doc_ids"] = doc_ids
        
        # 分页参数
        offset = (page - 1) * page_size
        limit = page_size
        
        # 构建MatchExpr列表
        matchExprs = []
        
        # 1. 文本搜索
        matchText, keywords = self.qryr.question(question, min_match=0.3)
        matchExprs.append(matchText)
        
        # 2. 向量搜索和融合（如果有向量模型）
        q_vec = []
        if emb_mdl:
            matchDense = self.get_vector(question, emb_mdl, self.config.top_k, self.config.similarity_threshold)
            q_vec = matchDense.embedding_data
            src.append(f"q_{len(q_vec)}_vec")
            
            # 创建融合表达式
            text_weight = 1.0 - self.config.vector_similarity_weight
            vector_weight = self.config.vector_similarity_weight
            fusionExpr = FusionExpr(
                "weighted_sum", 
                self.config.top_k, 
                {"weights": f"{text_weight:.2f}, {vector_weight:.2f}"}
            )
            
            matchExprs.extend([matchDense, fusionExpr])
        
        # 排序
        orderBy = OrderByExpr()
        
        # 执行搜索（直接使用索引名，无需ID转换）
        try:
            res = self.es_adapter.search(
                selectFields=src,
                highlightFields=highlightFields,
                condition=filters,
                matchExprs=matchExprs,
                orderBy=orderBy,
                offset=offset,
                limit=limit,
                indexNames=[self.config.index_name],  # 直接使用索引名
                knowledgebaseIds=[],  # 空的知识库ID
                aggFields=["docnm_kwd"]
            )
            
            # 构建结果对象
            class SearchResult:
                def __init__(self, es_result, es_adapter, keywords, query_vector):
                    self.total = es_adapter.getTotal(es_result)
                    self.ids = es_adapter.getChunkIds(es_result)
                    self.field = es_adapter.getFields(es_result, src)
                    self.highlight = es_adapter.getHighlight(es_result, keywords, "content_with_weight") if highlight else {}
                    self.aggregation = es_adapter.getAggregation(es_result, "docnm_kwd")
                    self.keywords = keywords
                    self.query_vector = query_vector
            
            return SearchResult(res, self.es_adapter, keywords, q_vec)
            
        except Exception as e:
            import traceback
            logger.error(f"搜索失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 返回空结果
            class EmptyResult:
                total = 0
                ids = []
                field = {}
                highlight = {}
                aggregation = {}
                keywords = []
                query_vector = []
            return EmptyResult()
    
    def rerank(self, sres, question: str, text_weight: float = 0.7, vector_weight: float = 0.3):
        """
        重排序方法（复用DeepRag的重排序逻辑）
        
        Args:
            sres: 搜索结果
            question: 查询问题
            text_weight: 文本权重
            vector_weight: 向量权重
            
        Returns:
            tuple: (综合相似度, 文本相似度, 向量相似度)
        """
        if sres.total == 0:
            return [], [], []
        
        # 获取查询的分词结果
        question_tokens = self.qryr.rmWWW(question).split()
        
        # 计算文本相似度
        text_similarities = []
        vector_similarities = []
        
        for chunk_id in sres.ids:
            chunk = sres.field[chunk_id]
            
            # 文本相似度计算
            content = chunk.get("content_ltks", "")
            content_tokens = self.qryr.rmWWW(content).split()
            
            # 使用混合相似度计算
            if hasattr(self.qryr, 'hybrid_similarity') and len(sres.query_vector) > 0:
                # 获取分块向量
                vector_field = f"q_{len(sres.query_vector)}_vec"
                chunk_vector = chunk.get(vector_field, [0.0] * len(sres.query_vector))
                
                # 计算混合相似度
                try:
                    sim, tsim, vsim = self.qryr.hybrid_similarity(
                        sres.query_vector, [chunk_vector], question_tokens, [content_tokens]
                    )
                    # 确保返回值是列表格式
                    if hasattr(tsim, '__iter__') and not isinstance(tsim, str):
                        text_similarities.append(float(tsim[0]) if len(tsim) > 0 else 0.0)
                    else:
                        text_similarities.append(float(tsim) if tsim else 0.0)

                    if hasattr(vsim, '__iter__') and not isinstance(vsim, str):
                        vector_similarities.append(float(vsim[0]) if len(vsim) > 0 else 0.0)
                    else:
                        vector_similarities.append(float(vsim) if vsim else 0.0)
                except Exception as e:
                    logger.warning(f"混合相似度计算失败，使用简单相似度: {e}")
                    # 回退到简单相似度
                    common_tokens = set(question_tokens) & set(content_tokens)
                    text_sim = len(common_tokens) / max(len(question_tokens), 1)
                    text_similarities.append(text_sim)
                    vector_similarities.append(0.0)
            else:
                # 简单的词汇重叠相似度
                common_tokens = set(question_tokens) & set(content_tokens)
                text_sim = len(common_tokens) / max(len(question_tokens), 1)
                text_similarities.append(text_sim)
                vector_similarities.append(0.0)
        
        # 计算综合相似度
        combined_similarities = []
        for i in range(len(text_similarities)):
            combined_sim = text_weight * text_similarities[i] + vector_weight * vector_similarities[i]
            combined_similarities.append(combined_sim)
        
        return combined_similarities, text_similarities, vector_similarities

    def retrieval(self, question: str, emb_mdl, page: int = 1, page_size: int = 10,
                 similarity_threshold: float = 0.2, vector_similarity_weight: float = 0.3,
                 top: int = 1024, doc_ids: List[str] = None, rerank_mdl=None, highlight: bool = True):
        """
        主要的检索方法（完全移除ID依赖，保持检索效果不变）

        Args:
            question: 查询问题
            emb_mdl: 向量化模型
            page: 页码
            page_size: 每页大小
            similarity_threshold: 相似度阈值
            vector_similarity_weight: 向量相似度权重
            top: 向量召回top-k
            doc_ids: 指定文档ID列表
            rerank_mdl: 重排序模型
            highlight: 是否高亮

        Returns:
            检索结果字典，格式与原版完全一致
        """
        if not question:
            return {"total": 0, "chunks": [], "doc_aggs": {}}

        logger.info(f"开始简化检索，问题: {question}")

        try:
            # 更新配置
            self.config.page = page
            self.config.page_size = page_size
            self.config.similarity_threshold = similarity_threshold
            self.config.vector_similarity_weight = vector_similarity_weight
            self.config.top_k = top
            self.config.highlight = highlight

            # 执行搜索
            sres = self.search(
                question=question,
                emb_mdl=emb_mdl,
                highlight=highlight,
                page=page if page > self.config.rerank_page_limit else 1,
                page_size=max(page_size * self.config.rerank_page_limit, 128) if page <= self.config.rerank_page_limit else page_size,
                doc_ids=doc_ids
            )

            if sres.total == 0:
                # 降级策略：降低相似度阈值
                logger.info("首次搜索无结果，尝试降级策略")
                self.config.similarity_threshold = 0.17
                # 重新创建查询器，使用更宽松的参数
                self.qryr = query.FulltextQueryer()
                sres = self.search(
                    question=question,
                    emb_mdl=emb_mdl,
                    highlight=highlight,
                    page=page if page > self.config.rerank_page_limit else 1,
                    page_size=max(page_size * self.config.rerank_page_limit, 128) if page <= self.config.rerank_page_limit else page_size,
                    doc_ids=doc_ids
                )

            # 构建返回结果
            ranks = {"total": sres.total, "chunks": [], "doc_aggs": sres.aggregation}

            if sres.total == 0:
                return ranks

            # 重排序处理
            if page <= self.config.rerank_page_limit:
                if rerank_mdl and sres.total > 0:
                    # 使用重排序模型
                    sim, tsim, vsim = self.rerank_by_model(rerank_mdl, sres, question,
                                                         1 - vector_similarity_weight, vector_similarity_weight)
                else:
                    # 使用默认重排序
                    sim, tsim, vsim = self.rerank(sres, question,
                                                1 - vector_similarity_weight, vector_similarity_weight)

                # 按相似度排序并分页
                # 确保sim是数值列表
                if isinstance(sim, dict):
                    sim = list(sim.values()) if sim else []
                sim_array = np.array(sim, dtype=float)
                idx = np.argsort(sim_array * -1)[(page - 1) * page_size:page * page_size]
            else:
                # 超过重排序页面限制，使用原始顺序
                sim = tsim = vsim = [1.0] * len(sres.ids)
                idx = list(range(len(sres.ids)))

            # 构建分块结果
            dim = len(sres.query_vector) if sres.query_vector else 0
            vector_column = f"q_{dim}_vec" if dim > 0 else None
            zero_vector = [0.0] * dim if dim > 0 else []

            for i in idx:
                if i >= len(sres.ids):
                    break
                if sim[i] < similarity_threshold:
                    break
                if len(ranks["chunks"]) >= page_size:
                    break

                chunk_id = sres.ids[i]
                chunk = sres.field[chunk_id]

                # 构建分块信息
                chunk_info = {
                    "id": chunk_id,
                    "content_with_weight": chunk.get("content_with_weight", ""),
                    "content_ltks": chunk.get("content_ltks", ""),
                    "docnm_kwd": chunk.get("docnm_kwd", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "img_id": chunk.get("img_id", ""),
                    "important_kwd": chunk.get("important_kwd", []),
                    "kb_id": "",  # 空的知识库ID
                    "page_num_int": chunk.get("page_num_int", []),
                    "position_int": chunk.get("position_int", 0),
                    "q_1024_vec": chunk.get(vector_column, zero_vector) if vector_column else zero_vector,
                    "similarity": float(sim[i]),
                    "term_similarity": float(tsim[i]) if i < len(tsim) else 0.0,
                    "vector_similarity": float(vsim[i]) if i < len(vsim) else 0.0,
                    "title_tks": chunk.get("title_tks", ""),
                    "top_int": chunk.get("top_int", 0)
                }

                # 添加高亮信息
                if highlight and chunk_id in sres.highlight:
                    chunk_info["highlight"] = sres.highlight[chunk_id]

                ranks["chunks"].append(chunk_info)

            logger.info(f"检索完成，返回 {len(ranks['chunks'])} 个分块")
            return ranks

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return {"total": 0, "chunks": [], "doc_aggs": {}}

    def rerank_by_model(self, rerank_mdl, sres, question: str, text_weight: float, vector_weight: float):
        """
        使用重排序模型进行重排序

        Args:
            rerank_mdl: 重排序模型
            sres: 搜索结果
            question: 查询问题
            text_weight: 文本权重
            vector_weight: 向量权重

        Returns:
            tuple: (综合相似度, 文本相似度, 向量相似度)
        """
        if sres.total == 0:
            return [], [], []

        try:
            # 准备重排序数据
            passages = []
            for chunk_id in sres.ids:
                chunk = sres.field[chunk_id]
                content = chunk.get("content_with_weight", "") or chunk.get("content_ltks", "")
                passages.append(content)

            # 使用重排序模型
            scores = rerank_mdl.similarity(question, passages)

            # 如果重排序模型返回的是numpy数组，转换为列表
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]

            # 确保分数数量与分块数量一致
            while len(scores) < len(sres.ids):
                scores.append(0.0)

            # 重排序模型的分数作为综合相似度
            sim = scores[:len(sres.ids)]

            # 计算文本和向量相似度（用于兼容性）
            tsim, vsim = self.rerank(sres, question, text_weight, vector_weight)[1:3]

            logger.debug(f"重排序完成，相似度范围: {np.min(sim):.4f} - {np.max(sim):.4f}")
            return sim, tsim, vsim

        except Exception as e:
            logger.error(f"重排序模型失败，回退到默认重排序: {e}")
            return self.rerank(sres, question, text_weight, vector_weight)

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查ES连接
            es_health = self.es_conn.get_health()
            es_status = es_health.get("status") in ["green", "yellow"]

            # 检查索引
            index_exists = self.es_conn.index_exists(self.config.index_name)

            return {
                "status": "healthy" if es_status and index_exists else "unhealthy",
                "elasticsearch": {
                    "status": es_health.get("status", "unknown"),
                    "cluster_name": es_health.get("cluster_name", "unknown")
                },
                "index": {
                    "name": self.config.index_name,
                    "exists": index_exists
                },
                "config": {
                    "es_host": self.config.es_host,
                    "similarity_threshold": self.config.similarity_threshold,
                    "vector_weight": self.config.vector_similarity_weight
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
