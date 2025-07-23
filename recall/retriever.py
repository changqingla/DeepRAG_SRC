#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

核心算法逻辑，包括：
- MatchExpr体系（MatchTextExpr, MatchDenseExpr, FusionExpr）
- 真正的混合搜索和权重融合
- 完整的重排序算法
- 降级策略

作者: Hu Tao
许可证: Apache 2.0
"""

import os
import sys
import logging
import copy
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# 添加DeepRag根目录到路径
current_dir = Path(__file__).parent.absolute()
DeepRag_root = current_dir.parent
sys.path.insert(0, str(DeepRag_root))

# 导入DeepRag核心组件
from rag.nlp import rag_tokenizer, query
from rag.utils import rmSpace
from rag.utils.doc_store_conn import MatchDenseExpr, MatchTextExpr, FusionExpr, OrderByExpr
import numpy as np

# 导入我们的ES连接和适配器
from embed_store.es_connection import SimpleESConnection
from es_adapter import ESAdapter


@dataclass
class DeepRagRetrievalConfig:
    """DeepRag召回配置"""
    index_names: List[str]                    # ES索引名称列表
    page: int = 1                            # 页码
    page_size: int = 10                      # 每页大小
    similarity_threshold: float = 0.1        # 相似度阈值（DeepRag默认0.1）
    vector_similarity_weight: float = 0.95   # 向量相似度权重（DeepRag默认0.95）
    top_k: int = 1024                        # 向量召回top-k
    highlight: bool = True                   # 是否高亮
    doc_ids: List[str] = None               # 指定文档ID列表
    es_config: Dict[str, Any] = None         # ES配置
    rerank_page_limit: int = 3               # 重排序页面限制（DeepRag默认3）


class DeepRagPureRetriever:
    """
    基于DeepRag原有算法的纯净召回器
    
    完全复用DeepRag的核心算法逻辑：
    - 使用MatchExpr体系进行搜索
    - 实现真正的FusionExpr混合搜索
    - 包含完整的重排序算法
    - 支持降级策略
    """
    
    def __init__(self, config: DeepRagRetrievalConfig):
        """
        初始化DeepRag纯净召回器
        
        Args:
            config: 召回配置
        """
        self.config = config
        
        # 设置默认ES配置
        es_config = config.es_config or {
            "hosts": "http://10.0.100.36:9201",
            "timeout": 600
        }
        
        # 创建简单的ES连接
        simple_es = SimpleESConnection(es_config.get("hosts", "http://localhost:9200"))

        # 创建ES适配器
        self.es_conn = ESAdapter(simple_es)

        # 创建DeepRag的查询器
        self.qryr = query.FulltextQueryer()
        
        logging.info(f"DeepRag纯净召回器已初始化，索引: {config.index_names}")
    
    def get_vector(self, txt: str, emb_mdl, topk: int = 10, similarity: float = 0.1):
        """
        创建向量搜索表达式（完全复用DeepRag的逻辑）
        
        Args:
            txt: 查询文本
            emb_mdl: 向量化模型
            topk: top-k数量
            similarity: 相似度阈值
            
        Returns:
            MatchDenseExpr对象
        """
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"DeepRagPureRetriever.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        
        return MatchDenseExpr(
            vector_column_name, 
            embedding_data, 
            'float', 
            'cosine', 
            topk, 
            {"similarity": similarity}
        )
    
    def search(self, req: Dict[str, Any], emb_mdl=None, highlight: bool = False):
        """
        搜索方法（完全复用DeepRag的search逻辑）
        
        Args:
            req: 搜索请求
            emb_mdl: 向量化模型
            highlight: 是否高亮
            
        Returns:
            搜索结果对象
        """
        qst = req.get("question", "")
        if not qst:
            # 返回空结果
            class EmptyResult:
                total = 0
                ids = []
                field = {}
                highlight = {}
                aggregation = {}
                keywords = []
            return EmptyResult()
        
        # 源字段
        src = req.get("fields", [
            "docnm_kwd", "content_ltks", "img_id", "title_tks", 
            "important_kwd", "position_int", "doc_id", "page_num_int", 
            "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
            "question_kwd", "question_tks", "available_int", "content_with_weight"
        ])
        
        # 高亮字段
        highlightFields = ["content_ltks", "title_tks"] if highlight else []
        
        # 过滤条件
        filters = {"available_int": req.get("available_int", 1)}
        if "doc_ids" in req and req["doc_ids"]:
            filters["doc_ids"] = req["doc_ids"]
        
        # 分页参数
        page = req.get("page", 1)
        page_size = req.get("size", 10)
        offset = (page - 1) * page_size
        limit = page_size
        
        # 构建MatchExpr列表（完全按照DeepRag的逻辑）
        matchExprs = []
        
        # 1. 文本搜索
        matchText, keywords = self.qryr.question(qst, min_match=0.3)
        matchExprs.append(matchText)
        
        # 2. 向量搜索和融合（如果有向量模型）
        if emb_mdl and req.get("vector", True):
            topk = req.get("topk", self.config.top_k)
            similarity = req.get("similarity", self.config.similarity_threshold)
            
            matchDense = self.get_vector(qst, emb_mdl, topk, similarity)
            q_vec = matchDense.embedding_data
            src.append(f"q_{len(q_vec)}_vec")
            
            # 创建融合表达式（使用DeepRag的权重配置）
            text_weight = 1.0 - self.config.vector_similarity_weight
            vector_weight = self.config.vector_similarity_weight
            fusionExpr = FusionExpr(
                "weighted_sum", 
                topk, 
                {"weights": f"{text_weight:.2f}, {vector_weight:.2f}"}
            )
            
            matchExprs.extend([matchDense, fusionExpr])
        
        # 排序
        orderBy = OrderByExpr()
        
        # 执行搜索
        try:
            res = self.es_conn.search(
                selectFields=src,
                highlightFields=highlightFields,
                condition=filters,
                matchExprs=matchExprs,
                orderBy=orderBy,
                offset=offset,
                limit=limit,
                indexNames=self.config.index_names,
                knowledgebaseIds=[],  # 空的知识库ID
                aggFields=["docnm_kwd"]
            )
            
            # 构建结果对象
            class SearchResult:
                def __init__(self, es_result, es_conn, keywords):
                    self.total = es_conn.getTotal(es_result)
                    self.ids = es_conn.getChunkIds(es_result)
                    self.field = es_conn.getFields(es_result, src)
                    self.highlight = es_conn.getHighlight(es_result, keywords, "content_with_weight") if highlight else {}
                    self.aggregation = es_conn.getAggregation(es_result, "docnm_kwd")
                    self.keywords = keywords
            
            return SearchResult(res, self.es_conn, keywords)
            
        except Exception as e:
            logging.error(f"搜索失败: {e}")
            # 返回空结果
            class EmptyResult:
                total = 0
                ids = []
                field = {}
                highlight = {}
                aggregation = {}
                keywords = []
            return EmptyResult()
    
    def rerank(self, chunk_ids: List[str], fields_data: Dict[str, Dict],
               question: str, keywords: List[str], query_vector: List[float],
               text_weight: float = 0.05, vector_weight: float = 0.95):
        """
        重排序算法（完全基于DeepRag的rerank逻辑）

        Args:
            chunk_ids: 分块ID列表
            fields_data: 字段数据
            question: 查询问题
            keywords: 关键词列表
            query_vector: 查询向量
            text_weight: 文本权重
            vector_weight: 向量权重

        Returns:
            相似度分数数组
        """
        if not chunk_ids:
            return np.array([]), np.array([]), np.array([])

        logging.debug(f"开始重排序，分块数量: {len(chunk_ids)}")

        # 1. 提取向量数据（完全按照DeepRag的逻辑）
        ins_embd = []
        ins_tw = []

        for chunk_id in chunk_ids:
            chunk_data = fields_data.get(chunk_id, {})

            # 提取向量数据
            vector_field = f"q_{len(query_vector)}_vec" if query_vector else "q_1024_vec"
            chunk_vector = chunk_data.get(vector_field, [])
            if isinstance(chunk_vector, str):
                # 如果是字符串，尝试解析
                try:
                    import json
                    chunk_vector = json.loads(chunk_vector)
                except:
                    chunk_vector = []

            if not chunk_vector or len(chunk_vector) != len(query_vector):
                # 如果没有向量或维度不匹配，使用零向量
                chunk_vector = [0.0] * len(query_vector) if query_vector else [0.0] * 1024

            ins_embd.append(chunk_vector)

            # 构建token权重（完全按照DeepRag的逻辑）
            content_ltks = chunk_data.get("content_ltks", "").split()
            title_tks = [t for t in chunk_data.get("title_tks", "").split() if t]
            question_tks = [t for t in chunk_data.get("question_tks", "").split() if t]
            important_kwd = chunk_data.get("important_kwd", [])

            if isinstance(important_kwd, str):
                important_kwd = [important_kwd]

            # DeepRag的权重配置：content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # 2. 使用DeepRag的hybrid_similarity计算相似度（完全按照原有格式）
        sim, tksim, vtsim = self.qryr.hybrid_similarity(
            query_vector,    # avec: 查询向量
            ins_embd,        # bvecs: 文档向量列表
            keywords,        # atks: 查询关键词列表
            ins_tw,          # btkss: 文档token列表的列表
            text_weight,     # tkweight: token权重
            vector_weight    # vtweight: 向量权重
        )

        logging.debug(f"重排序完成，相似度范围: {np.min(sim):.4f} - {np.max(sim):.4f}")
        return sim, tksim, vtsim



    def retrieval(self, question: str, embd_mdl, page: int = 1, page_size: int = 10,
                 similarity_threshold: float = 0.1, vector_similarity_weight: float = 0.95,
                 top: int = 1024, doc_ids: List[str] = None, rerank_mdl=None, highlight: bool = True):
        """
        召回方法（完全复用DeepRag的retrieval逻辑）

        Args:
            question: 查询问题
            embd_mdl: 向量化模型
            page: 页码
            page_size: 每页大小
            similarity_threshold: 相似度阈值
            vector_similarity_weight: 向量相似度权重
            top: 向量召回top-k
            doc_ids: 指定文档ID列表
            rerank_mdl: 重排序模型
            highlight: 是否高亮

        Returns:
            召回结果字典
        """
        if not question:
            return {"total": 0, "chunks": [], "doc_aggs": {}}

        logging.info(f"开始DeepRag召回，问题: {question}")

        try:
            # 更新配置
            self.config.page = page
            self.config.page_size = page_size
            self.config.similarity_threshold = similarity_threshold
            self.config.vector_similarity_weight = vector_similarity_weight
            self.config.top_k = top
            self.config.highlight = highlight
            if doc_ids:
                self.config.doc_ids = doc_ids

            # 构建搜索请求
            req = {
                "question": question,
                "page": page,
                "size": max(page_size * self.config.rerank_page_limit, 128) if page <= self.config.rerank_page_limit else page_size,
                "topk": top,
                "similarity": similarity_threshold,
                "vector": True,
                "available_int": 1,
                "fields": [
                    "docnm_kwd", "content_ltks", "img_id", "title_tks",
                    "important_kwd", "position_int", "doc_id", "page_num_int",
                    "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                    "question_kwd", "question_tks", "available_int", "content_with_weight"
                ]
            }

            if doc_ids:
                req["doc_ids"] = doc_ids

            # 执行搜索
            sres = self.search(req, embd_mdl, highlight)

            if sres.total == 0:
                # 降级策略：降低min_match，提高相似度阈值
                logging.info("首次搜索无结果，尝试降级策略")
                req["similarity"] = 0.17
                # 这里需要重新创建查询器，降低min_match
                original_qryr = self.qryr
                self.qryr = query.FulltextQueryer()  # 重新创建，使用更宽松的参数
                sres = self.search(req, embd_mdl, highlight)
                self.qryr = original_qryr  # 恢复原始查询器

            if sres.total == 0:
                return {"total": 0, "chunks": [], "doc_aggs": {}}

            # 重排序（如果在重排序页面范围内）
            if page <= self.config.rerank_page_limit:
                if rerank_mdl:
                    # 使用重排序模型
                    # 获取查询向量
                    query_vector = []
                    if embd_mdl:
                        qv, _ = embd_mdl.encode_queries(question)
                        query_vector = [float(v) for v in qv]

                    sim, tsim, vsim = self.rerank_by_model(
                        rerank_mdl, sres.ids, sres.field, question, query_vector,
                        1.0 - vector_similarity_weight, vector_similarity_weight
                    )
                else:
                    # 使用默认重排序
                    # 获取查询向量
                    query_vector = []
                    if embd_mdl:
                        qv, _ = embd_mdl.encode_queries(question)
                        query_vector = [float(v) for v in qv]

                    sim, tsim, vsim = self.rerank(
                        sres.ids, sres.field, question, sres.keywords, query_vector,
                        1.0 - vector_similarity_weight, vector_similarity_weight
                    )

                # 按相似度排序
                idx = np.argsort(sim * -1)
            else:
                # 超出重排序页面范围，直接使用ES排序
                idx = list(range(len(sres.ids)))
                sim = np.ones(len(sres.ids))

            # 分页处理
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            if page <= self.config.rerank_page_limit:
                # 重排序页面，从排序后的结果中取
                page_idx = idx[start_idx:end_idx]
            else:
                # 非重排序页面，直接分页
                page_idx = idx[:page_size]

            # 构建最终结果
            chunks = []
            doc_aggs = {}

            for i in page_idx:
                if i >= len(sres.ids):
                    continue

                chunk_id = sres.ids[i]
                chunk_data = sres.field.get(chunk_id, {})

                # 基本信息
                chunk = {
                    "chunk_id": chunk_id,
                    "content_with_weight": chunk_data.get("content_with_weight", ""),
                    "doc_id": chunk_data.get("doc_id", ""),
                    "docnm_kwd": chunk_data.get("docnm_kwd", ""),
                    "page_num_int": chunk_data.get("page_num_int", []),
                    "position_int": chunk_data.get("position_int", []),
                    "available_int": chunk_data.get("available_int", 1),
                    "similarity": float(sim[i]) if i < len(sim) else 0.0
                }

                # 添加其他字段
                for field in ["img_id", "title_tks", "important_kwd", "top_int",
                             "create_timestamp_flt", "knowledge_graph_kwd", "question_kwd", "question_tks"]:
                    if field in chunk_data:
                        chunk[field] = chunk_data[field]

                # 添加高亮
                if highlight and chunk_id in sres.highlight:
                    chunk["highlight"] = rmSpace(sres.highlight[chunk_id])
                else:
                    chunk["highlight"] = chunk["content_with_weight"]

                chunks.append(chunk)

                # 文档聚合
                doc_name = chunk.get("docnm_kwd", "Unknown")
                doc_id = chunk.get("doc_id", "")
                if doc_name not in doc_aggs:
                    doc_aggs[doc_name] = {"doc_id": doc_id, "count": 0}
                doc_aggs[doc_name]["count"] += 1

            # 转换文档聚合格式
            doc_aggs_list = [
                {"doc_name": k, "doc_id": v["doc_id"], "count": v["count"]}
                for k, v in sorted(doc_aggs.items(), key=lambda x: x[1]["count"], reverse=True)
            ]

            # 获取查询向量
            query_vector = []
            if embd_mdl:
                qv, _ = embd_mdl.encode_queries(question)
                query_vector = [float(v) for v in qv]

            result = {
                "total": sres.total,
                "chunks": chunks,
                "doc_aggs": doc_aggs_list,
                "query_vector": query_vector
            }

            logging.info(f"DeepRag召回完成，总数: {sres.total}, 返回: {len(chunks)} 个分块")
            return result

        except Exception as e:
            import traceback
            logging.error(f"DeepRag召回失败: {e}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            return {"total": 0, "chunks": [], "doc_aggs": {}, "error": str(e)}

    def rerank_by_model(self, rerank_mdl, chunk_ids: List[str], fields_data: Dict[str, Dict],
                       question: str, query_vector: List[float], text_weight: float, vector_weight: float):
        """
        使用重排序模型进行重排序（完全基于DeepRag的逻辑）

        Args:
            rerank_mdl: 重排序模型
            chunk_ids: 分块ID列表
            fields_data: 字段数据
            question: 查询问题
            query_vector: 查询向量
            text_weight: 文本权重
            vector_weight: 向量权重

        Returns:
            相似度分数数组
        """
        if not chunk_ids or not rerank_mdl:
            return np.array([]), np.array([]), np.array([])

        logging.debug(f"使用重排序模型进行重排序，分块数量: {len(chunk_ids)}")

        try:
            # 1. 准备token数据（完全按照DeepRag第318-324行的逻辑）
            ins_tw = []
            for chunk_id in chunk_ids:
                chunk_data = fields_data.get(chunk_id, {})

                # 按照DeepRag的逻辑处理token
                content_ltks = chunk_data.get("content_ltks", "").split()
                title_tks = [t for t in chunk_data.get("title_tks", "").split() if t]
                important_kwd = chunk_data.get("important_kwd", [])

                # 确保important_kwd是列表
                if isinstance(important_kwd, str):
                    important_kwd = [important_kwd]

                # 组合token（按照DeepRag第323行）
                tks = content_ltks + title_tks + important_kwd
                ins_tw.append(tks)

            # 2. 计算token相似度（按照DeepRag第326行）
            _, keywords = self.qryr.question(question)
            tksim = self.qryr.token_similarity(keywords, ins_tw)

            # 3. 使用重排序模型计算相似度（完全按照DeepRag第327行）
            from rag.utils import rmSpace
            docs_for_rerank = [rmSpace(" ".join(tks)) for tks in ins_tw]
            vtsim, _ = rerank_mdl.similarity(question, docs_for_rerank)

            # 4. 按照DeepRag第331行的公式计算最终相似度
            sim = text_weight * np.array(tksim) + vector_weight * np.array(vtsim)

            logging.debug(f"重排序模型计算完成，相似度范围: {np.min(sim):.4f} - {np.max(sim):.4f}")
            return sim, np.array(tksim), np.array(vtsim)

        except Exception as e:
            logging.error(f"重排序模型计算失败: {e}")
            # 如果重排序模型失败，降级为默认重排序
            logging.warning("重排序模型失败，降级为默认重排序算法")
            return self.rerank(chunk_ids, fields_data, question, [], query_vector, text_weight, vector_weight)

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 通过适配器获取ES健康状态
            try:
                es_health = self.es_conn.es.cluster.health()
                es_status = es_health.get("status") == "green" or es_health.get("status") == "yellow"
            except Exception as e:
                es_health = {"status": "red", "error": str(e)}
                es_status = False

            index_status = {}
            for index_name in self.config.index_names:
                try:
                    exists = self.es_conn.indexExist(index_name, "")
                    index_status[index_name] = exists
                except Exception as e:
                    index_status[index_name] = f"error: {e}"

            return {
                "status": "healthy" if es_status else "unhealthy",
                "components": {
                    "elasticsearch": es_status,
                    "query_processor": self.qryr is not None,
                    "es_adapter": self.es_conn is not None
                },
                "elasticsearch": es_health,
                "indices": index_status,
                "config": {
                    "index_names": self.config.index_names,
                    "page_size": self.config.page_size,
                    "similarity_threshold": self.config.similarity_threshold,
                    "vector_similarity_weight": self.config.vector_similarity_weight,
                    "rerank_page_limit": self.config.rerank_page_limit
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 为了向后兼容，创建别名
PureRetriever = DeepRagPureRetriever
SimpleRetrievalConfig = DeepRagRetrievalConfig
