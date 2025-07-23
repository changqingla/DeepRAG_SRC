#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重写的Elasticsearch适配器

将新的SimpleESConnection适配为DeepRAG的DocStoreConnection接口。

作者: Hu Tao
许可证: Apache 2.0
"""

import os
import sys
import logging
import json
import copy
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加DeepRAG根目录到路径
current_dir = Path(__file__).parent.absolute()
deeprag_root = current_dir.parent
sys.path.insert(0, str(deeprag_root))

# 导入DeepRAG核心组件
from rag.utils.doc_store_conn import DocStoreConnection, MatchExpr, OrderByExpr
from rag.utils.doc_store_conn import MatchTextExpr, MatchDenseExpr, FusionExpr
from rag.utils import rmSpace

# 导入我们重写的ES连接
from embed_store.es_connection import SimpleESConnection

logger = logging.getLogger('recall.es_adapter')


class ESAdapter(DocStoreConnection):
    """
    重写的Elasticsearch适配器

    将新的SimpleESConnection适配为DeepRAG的DocStoreConnection接口。
    实现DocStoreConnection的所有抽象方法。
    """

    def __init__(self, es_conn: SimpleESConnection):
        """
        初始化适配器

        Args:
            es_conn: SimpleESConnection实例
        """
        self.es_conn = es_conn
        self.es = es_conn.es
        logger.info("重写的Elasticsearch适配器已初始化")
    
    def indexExist(self, indexName: str, knowledgebaseId: str = "") -> bool:
        """
        检查索引是否存在

        Args:
            indexName: 索引名称
            knowledgebaseId: 知识库ID（兼容参数，不使用）

        Returns:
            索引是否存在
        """
        try:
            return self.es_conn.index_exists(indexName)
        except Exception as e:
            logger.error(f"检查索引失败: {e}")
            return False
    
    def search(self, selectFields: list[str], highlightFields: list[str],
              condition: dict, matchExprs: list[MatchExpr],
              orderBy: OrderByExpr, offset: int, limit: int,
              indexNames: str|list[str], knowledgebaseIds: list[str],
              aggFields: list[str] = [], rank_feature: dict | None = None):
        """
        搜索方法（实现DeepRAG的DocStoreConnection接口）

        Args:
            selectFields: 选择字段
            highlightFields: 高亮字段
            condition: 过滤条件
            matchExprs: 匹配表达式列表
            orderBy: 排序表达式
            offset: 偏移量
            limit: 限制数量
            indexNames: 索引名称
            knowledgebaseIds: 知识库ID列表（兼容参数，不使用）
            aggFields: 聚合字段
            rank_feature: 排序特征（兼容参数，不使用）

        Returns:
            搜索结果
        """
        # 标准化索引名称
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0

        # 构建ES查询
        es_query = self._build_es_query(
            selectFields, highlightFields, condition,
            matchExprs, orderBy, offset, limit,
            aggFields, indexNames
        )

        # 执行搜索（简化重试逻辑）
        try:
            # 使用新的SimpleESConnection的search方法
            res = self.es_conn.search(
                index_name=indexNames[0] if len(indexNames) == 1 else ",".join(indexNames),
                query=es_query,
                size=limit if limit > 0 else 10
            )

            # 转换为标准ES响应格式
            if "hits" not in res:
                # 如果返回格式不标准，构造标准格式
                res = {
                    "hits": {
                        "hits": res.get("hits", []),
                        "total": {"value": len(res.get("hits", []))}
                    }
                }

            return res

        except Exception as e:
            logger.error(f"搜索失败: {e}")

            # 如果是向量字段相关错误，尝试降级为纯文本搜索
            if "unknown field" in str(e) and "vec" in str(e):
                logger.warning("向量字段不存在，尝试纯文本搜索")
                try:
                    fallback_query = self._build_fallback_query(
                        selectFields, highlightFields, condition,
                        matchExprs, orderBy, offset, limit, aggFields
                    )

                    res = self.es_conn.search(
                        index_name=indexNames[0] if len(indexNames) == 1 else ",".join(indexNames),
                        query=fallback_query,
                        size=limit if limit > 0 else 10
                    )

                    return res

                except Exception as fallback_e:
                    logger.error(f"降级搜索也失败: {fallback_e}")
                    raise e

            raise e
    
    def _build_es_query(self, selectFields, highlightFields, condition,
                       matchExprs, orderBy, offset, limit,
                       aggFields, index_names):
        """
        构建ES查询（简化版，适配新的SimpleESConnection）

        Args:
            selectFields: 选择字段
            highlightFields: 高亮字段
            condition: 过滤条件
            matchExprs: 匹配表达式列表
            orderBy: 排序表达式
            offset: 偏移量
            limit: 限制数量
            aggFields: 聚合字段
            index_names: 索引名称列表

        Returns:
            ES查询字典
        """
        # 构建基础bool查询
        bool_query = {
            "bool": {
                "must": [],
                "should": [],
                "filter": []
            }
        }

        # 添加过滤条件
        if condition:
            for k, v in condition.items():
                if k == "doc_ids" and v:
                    bool_query["bool"]["filter"].append({"terms": {"doc_id": v}})
                elif k == "available_int":
                    bool_query["bool"]["filter"].append({"term": {"available_int": v}})
                elif k == "docnm_kwd" and v:
                    bool_query["bool"]["filter"].append({"term": {"docnm_kwd": v}})
                else:
                    bool_query["bool"]["filter"].append({"term": {k: v}})

        # 构建查询结构
        query = {"query": bool_query}

        # 添加高亮
        if highlightFields:
            query["highlight"] = {
                "fields": {field: {} for field in highlightFields}
            }

        # 添加源字段
        if selectFields:
            query["_source"] = selectFields
        
        # 处理匹配表达式
        vector_similarity_weight = 0.5
        has_vector_search = False
        vector_expr = None

        # 首先检查是否有向量搜索和融合表达式
        for m in matchExprs:
            if isinstance(m, FusionExpr) and m.method == "weighted_sum" and "weights" in m.fusion_params:
                weights = m.fusion_params["weights"]
                vector_similarity_weight = float(weights.split(",")[1])
            elif isinstance(m, MatchDenseExpr):
                vector_expr = m
                has_vector_search = True

        # 处理文本搜索
        for m in matchExprs:
            if isinstance(m, MatchTextExpr):
                minimum_should_match = m.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    minimum_should_match = str(int(minimum_should_match * 100)) + "%"

                bool_query["bool"]["must"].append({
                    "query_string": {
                        "fields": m.fields,
                        "type": "best_fields",
                        "query": m.matching_text,
                        "minimum_should_match": minimum_should_match,
                        "boost": 1.0 - vector_similarity_weight
                    }
                })

        # 处理向量搜索
        if has_vector_search and vector_expr:
            # 检查向量字段是否存在
            vector_field_exists = self._check_vector_field_exists(index_names, vector_expr.vector_column_name)

            if vector_field_exists:
                # 使用KNN查询（ES 8.0+ 语法）
                # KNN查询需要在顶层，不能在query中
                knn_query = {
                    "field": vector_expr.vector_column_name,
                    "query_vector": list(vector_expr.embedding_data),
                    "k": vector_expr.topn,
                    "num_candidates": vector_expr.topn * 2,
                }

                # 添加过滤条件到KNN查询
                if bool_query["bool"]["filter"] or bool_query["bool"]["must"]:
                    knn_query["filter"] = bool_query

                # 添加相似度阈值
                if "similarity" in vector_expr.extra_options:
                    knn_query["similarity"] = vector_expr.extra_options["similarity"]

                # 构建完整的查询体
                query = {
                    "knn": knn_query,
                    "_source": selectFields,
                    "highlight": query.get("highlight", {})
                }
            else:
                # 向量字段不存在，降级为纯文本搜索
                logger.warning(f"向量字段 {vector_expr.vector_column_name} 不存在，降级为纯文本搜索")
                query["query"] = bool_query
        else:
            # 没有向量搜索，只使用文本搜索
            query["query"] = bool_query

        # 添加排序
        if orderBy and orderBy.fields:
            query["sort"] = []
            for field, order in orderBy.fields:
                order_str = "asc" if order == 0 else "desc"
                if field in ["page_num_int", "top_int"]:
                    query["sort"].append({
                        field: {
                            "order": order_str,
                            "unmapped_type": "float",
                            "mode": "avg",
                            "numeric_type": "double"
                        }
                    })
                elif field.endswith("_int") or field.endswith("_flt"):
                    query["sort"].append({
                        field: {
                            "order": order_str,
                            "unmapped_type": "float"
                        }
                    })
                else:
                    query["sort"].append({
                        field: {
                            "order": order_str,
                            "unmapped_type": "text"
                        }
                    })

        # 添加聚合
        if aggFields:
            query["aggs"] = {}
            for fld in aggFields:
                query["aggs"][f'aggs_{fld}'] = {
                    "terms": {
                        "field": fld,
                        "size": 1000000
                    }
                }

        # 添加分页
        if limit > 0:
            query["from"] = offset
            query["size"] = limit

        # 调试：打印生成的查询
        logger.debug(f"生成的ES查询: {json.dumps(query, indent=2)}")

        return query
    
    def getTotal(self, res):
        """获取总数"""
        return res.get("hits", {}).get("total", {}).get("value", 0)
    
    def getChunkIds(self, res):
        """获取分块ID列表"""
        return [hit["_id"] for hit in res.get("hits", {}).get("hits", [])]
    
    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        """获取字段数据"""
        res_fields = {}
        
        for hit in res.get("hits", {}).get("hits", []):
            d = hit["_source"]
            d["id"] = hit["_id"]
            
            m = {n: d.get(n) for n in fields if d.get(n) is not None}
            for n, v in m.items():
                if isinstance(v, list):
                    m[n] = v
                    continue
                # 保持available_int字段的原始整数类型
                if n == "available_int" and isinstance(v, (int, float)):
                    m[n] = v
                    continue
                if not isinstance(v, str):
                    m[n] = str(m[n])
            
            if m:
                res_fields[d["id"]] = m
        
        return res_fields
    
    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        """获取高亮数据"""
        # 兼容参数，实际不使用keywords和fieldnm
        _ = keywords, fieldnm

        highlights = {}

        for hit in res.get("hits", {}).get("hits", []):
            if "highlight" in hit:
                highlight_text = ""
                for _, highlights_list in hit["highlight"].items():
                    highlight_text += " ".join(highlights_list)

                highlights[hit["_id"]] = rmSpace(highlight_text)

        return highlights
    
    def getAggregation(self, res, field: str):
        """获取聚合数据"""
        if "aggregations" not in res:
            return []
        
        agg_key = f"aggs_{field}"
        if agg_key not in res["aggregations"]:
            return []
        
        return res["aggregations"][agg_key].get("buckets", [])
    
    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str] = None):
        """获取单个文档"""
        # 兼容参数，实际不使用knowledgebaseIds
        _ = knowledgebaseIds

        try:
            result = self.es.get(index=indexName, id=chunkId)
            if result["found"]:
                chunk = result["_source"]
                chunk["id"] = chunkId
                return chunk
            return None
        except Exception as e:
            logger.error(f"获取文档失败 {chunkId}: {e}")
            return None

    # 实现其他抽象方法（简化实现）
    def dbType(self) -> str:
        """返回数据库类型"""
        return "elasticsearch"

    def health(self):
        """健康检查"""
        try:
            return self.es.cluster.health()
        except Exception as e:
            return {"status": "red", "error": str(e)}

    def createIdx(self, indexName: str, knowledgebaseId: str, schema: dict):
        """创建索引"""
        # 兼容参数，实际不使用knowledgebaseId
        _ = knowledgebaseId

        try:
            # 使用新的SimpleESConnection方法
            # 从schema中提取向量维度
            vector_dim = 1024  # 默认维度
            if "mappings" in schema and "properties" in schema["mappings"]:
                for _, field_config in schema["mappings"]["properties"].items():
                    if field_config.get("type") == "dense_vector":
                        vector_dim = field_config.get("dims", 1024)
                        break

            return self.es_conn.create_index(indexName, vector_dim)
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        """删除索引"""
        # 兼容参数，实际不使用knowledgebaseId
        _ = knowledgebaseId

        try:
            return self.es_conn.delete_index(indexName)
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            return False

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        """插入文档"""
        # 兼容参数，实际不使用knowledgebaseId
        _ = knowledgebaseId

        try:
            result = self.es_conn.bulk_index(indexName, documents)
            # 返回成功插入的文档ID列表
            return [doc.get("id", str(i)) for i, doc in enumerate(documents[:result["success"]])]
        except Exception as e:
            logger.error(f"插入文档失败: {e}")
            return []

    def update(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> int:
        """更新文档"""
        # 兼容参数，实际不使用knowledgebaseId
        _ = knowledgebaseId

        try:
            # 简化实现：使用bulk_index进行更新
            result = self.es_conn.bulk_index(indexName, documents)
            return result["success"]
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return 0

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        """删除文档"""
        # 兼容参数，实际不使用knowledgebaseId
        _ = knowledgebaseId

        try:
            # 简化实现：通过查询删除
            query = {"query": {"bool": {"filter": []}}}
            for k, v in condition.items():
                query["query"]["bool"]["filter"].append({"term": {k: v}})

            # 使用ES的delete_by_query API
            result = self.es.delete_by_query(index=indexName, body=query)
            return result.get("deleted", 0)
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return 0

    def sql(self, sql: str, fetch_size: int, format: str):
        """执行SQL查询"""
        # 兼容参数，实际不使用
        _ = sql, fetch_size, format

        try:
            # 简化实现：不支持SQL查询
            logger.warning("SQL查询功能未实现")
            return None
        except Exception as e:
            logger.error(f"SQL查询失败: {e}")
            return None

    def _build_fallback_query(self, selectFields, highlightFields, condition,
                           matchExprs, orderBy, offset, limit, aggFields):
        """
        构建降级查询（仅使用文本搜索）

        Args:
            selectFields: 选择字段
            highlightFields: 高亮字段
            condition: 过滤条件
            matchExprs: 匹配表达式列表
            orderBy: 排序表达式
            offset: 偏移量
            limit: 限制数量
            aggFields: 聚合字段

        Returns:
            降级查询字典
        """
        # 重用主查询构建逻辑，但过滤掉向量搜索
        text_match_exprs = [m for m in matchExprs if isinstance(m, MatchTextExpr)]

        return self._build_es_query(
            selectFields, highlightFields, condition,
            text_match_exprs, orderBy, offset, limit,
            aggFields, []  # 空的index_names，因为这是降级查询
        )

    def _check_vector_field_exists(self, index_names, vector_field: str) -> bool:
        """
        检查向量字段是否存在

        Args:
            index_names: 索引名称
            vector_field: 向量字段名称

        Returns:
            字段是否存在
        """
        try:
            # 将字符串索引转换为列表
            if isinstance(index_names, str):
                index_names = [index_names]

            # 检查每个索引
            for index_name in index_names:
                try:
                    # 获取索引映射
                    mapping = self.es.indices.get_mapping(index=index_name)

                    # 检查字段是否存在
                    if index_name in mapping:
                        properties = mapping[index_name].get("mappings", {}).get("properties", {})
                        if vector_field in properties:
                            # 检查字段类型是否为向量
                            field_type = properties[vector_field].get("type")
                            if field_type in ["dense_vector", "knn_vector"]:
                                return True
                except Exception as e:
                    logging.warning(f"检查索引 {index_name} 映射失败: {e}")
                    continue

            return False
        except Exception as e:
            logging.warning(f"检查向量字段失败: {e}")
            return False
