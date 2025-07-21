#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elasticsearch适配器

将IndependentESConnection适配为deeprag的DocStoreConnection接口。

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
from dataclasses import dataclass

# 添加deeprag根目录到路径
current_dir = Path(__file__).parent.absolute()
deeprag_root = current_dir.parent
sys.path.insert(0, str(deeprag_root))

# 导入deeprag核心组件
from rag.utils.doc_store_conn import DocStoreConnection, MatchExpr, OrderByExpr
from rag.utils.doc_store_conn import MatchTextExpr, MatchDenseExpr, FusionExpr
from rag.utils import rmSpace

# 导入我们的ES连接
from embed_store.es_connection import IndependentESConnection


class ESAdapter(DocStoreConnection):
    """
    Elasticsearch适配器

    将IndependentESConnection适配为deeprag的DocStoreConnection接口。
    实现DocStoreConnection的所有抽象方法。
    """
    
    def __init__(self, es_conn: IndependentESConnection):
        """
        初始化适配器
        
        Args:
            es_conn: IndependentESConnection实例
        """
        self.es_conn = es_conn
        self.es = es_conn.es
        logging.info("Elasticsearch适配器已初始化")
    
    def indexExist(self, indexName: str, knowledgebaseId: str = "") -> bool:
        """
        检查索引是否存在
        
        Args:
            indexName: 索引名称
            knowledgebaseId: 知识库ID（不使用）
            
        Returns:
            索引是否存在
        """
        try:
            return self.es_conn.indexExist(indexName, "")
        except Exception as e:
            logging.error(f"检查索引失败: {e}")
            return False
    
    def search(self, selectFields: list[str], highlightFields: list[str],
              condition: dict, matchExprs: list[MatchExpr],
              orderBy: OrderByExpr, offset: int, limit: int,
              indexNames: str|list[str], knowledgebaseIds: list[str],
              aggFields: list[str] = [], rank_feature: dict | None = None):
        """
        搜索方法（实现deeprag的DocStoreConnection接口）
        
        Args:
            selectFields: 选择字段
            highlightFields: 高亮字段
            condition: 过滤条件
            matchExprs: 匹配表达式列表
            orderBy: 排序表达式
            offset: 偏移量
            limit: 限制数量
            indexNames: 索引名称
            knowledgebaseIds: 知识库ID列表（不使用）
            aggFields: 聚合字段
            rank_feature: 排序特征
            
        Returns:
            搜索结果
        """
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0
        
        # 构建ES查询
        es_query = self._build_es_query(
            selectFields, highlightFields, condition,
            matchExprs, orderBy, offset, limit,
            aggFields, rank_feature, indexNames
        )
        
        # 执行搜索
        for i in range(3):  # 重试3次
            try:
                res = self.es.search(
                    index=indexNames,
                    body=es_query,
                    timeout="600s",
                    track_total_hits=True,
                    _source=True
                )
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                return res
            except Exception as e:
                logging.error(f"搜索失败 (尝试 {i+1}/3): {e}")
                if str(e).find("Timeout") > 0:
                    continue
                # 如果是向量字段不存在的错误，尝试降级为纯文本搜索
                if "unknown field" in str(e) and "vec" in str(e):
                    logging.warning("向量字段不存在，尝试纯文本搜索")
                    # 重新构建查询，只使用文本搜索
                    fallback_query = self._build_fallback_query(
                        selectFields, highlightFields, condition,
                        matchExprs, orderBy, offset, limit, aggFields
                    )
                    try:
                        res = self.es.search(
                            index=indexNames,
                            body=fallback_query,
                            timeout="600s",
                            track_total_hits=True,
                            _source=True
                        )
                        return res
                    except Exception as fallback_e:
                        logging.error(f"降级搜索也失败: {fallback_e}")
                        raise e
                raise e

        logging.error("搜索超时3次!")
        raise Exception("搜索超时.")
    
    def _build_es_query(self, selectFields, highlightFields, condition,
                       matchExprs, orderBy, offset, limit,
                       aggFields, rank_feature, index_names):
        """
        构建ES查询（基于deeprag的逻辑）
        
        Args:
            selectFields: 选择字段
            highlightFields: 高亮字段
            condition: 过滤条件
            matchExprs: 匹配表达式列表
            orderBy: 排序表达式
            offset: 偏移量
            limit: 限制数量
            aggFields: 聚合字段
            rank_feature: 排序特征
            
        Returns:
            ES查询字典
        """
        # 构建bool查询
        bqry = {
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
                    bqry["bool"]["filter"].append({"terms": {"doc_id": v}})
                elif k == "available_int":
                    bqry["bool"]["filter"].append({"term": {"available_int": v}})
                else:
                    bqry["bool"]["filter"].append({"term": {k: v}})
        
        # 构建查询
        s = {"query": bqry}
        
        # 添加高亮
        if highlightFields:
            s["highlight"] = {"fields": {}}
            for field in highlightFields:
                s["highlight"]["fields"][field] = {}
        
        # 添加源字段
        if selectFields:
            s["_source"] = selectFields
        
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

                bqry["bool"]["must"].append({
                    "query_string": {
                        "fields": m.fields,
                        "type": "best_fields",
                        "query": m.matching_text,
                        "minimum_should_match": minimum_should_match,
                        "boost": 1.0 - vector_similarity_weight  # boost应该在查询子句内部
                    }
                })

        # 处理向量搜索
        if has_vector_search and vector_expr:
            # 检查向量字段是否存在
            vector_field_exists = self._check_vector_field_exists(index_names, vector_expr.vector_column_name)

            if vector_field_exists:
                # 使用KNN查询（ES 8.0+ 语法）
                s = {
                    "knn": {
                        "field": vector_expr.vector_column_name,
                        "query_vector": list(vector_expr.embedding_data),
                        "k": vector_expr.topn,
                        "num_candidates": vector_expr.topn * 2,
                    },
                    "_source": selectFields,
                    "highlight": s.get("highlight", {})
                }

                # 添加过滤条件到KNN查询
                if bqry["bool"]["filter"] or bqry["bool"]["must"]:
                    s["knn"]["filter"] = bqry

                # 添加相似度阈值
                if "similarity" in vector_expr.extra_options:
                    s["knn"]["similarity"] = vector_expr.extra_options["similarity"]
            else:
                # 向量字段不存在，降级为纯文本搜索
                logging.warning(f"向量字段 {vector_expr.vector_column_name} 不存在，降级为纯文本搜索")
                s["query"] = bqry
        else:
            # 没有向量搜索，只使用文本搜索
            s["query"] = bqry
        
        # 添加排序
        if orderBy and orderBy.fields:
            s["sort"] = []
            for field, order in orderBy.fields:
                order_str = "asc" if order == 0 else "desc"
                if field in ["page_num_int", "top_int"]:
                    s["sort"].append({
                        field: {
                            "order": order_str,
                            "unmapped_type": "float",
                            "mode": "avg",
                            "numeric_type": "double"
                        }
                    })
                elif field.endswith("_int") or field.endswith("_flt"):
                    s["sort"].append({
                        field: {
                            "order": order_str,
                            "unmapped_type": "float"
                        }
                    })
                else:
                    s["sort"].append({
                        field: {
                            "order": order_str,
                            "unmapped_type": "text"
                        }
                    })
        
        # 添加聚合
        if aggFields:
            s["aggs"] = {}
            for fld in aggFields:
                s["aggs"][f'aggs_{fld}'] = {
                    "terms": {
                        "field": fld,
                        "size": 1000000
                    }
                }
        
        # 添加分页
        if limit > 0:
            s["from"] = offset
            s["size"] = limit

        # 调试：打印生成的查询
        logging.debug(f"生成的ES查询: {json.dumps(s, indent=2)}")

        return s
    
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
        highlights = {}
        
        for hit in res.get("hits", {}).get("hits", []):
            if "highlight" in hit:
                highlight_text = ""
                for field, highlights_list in hit["highlight"].items():
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
        try:
            result = self.es.get(index=indexName, id=chunkId)
            if result["found"]:
                chunk = result["_source"]
                chunk["id"] = chunkId
                return chunk
            return None
        except Exception as e:
            logging.error(f"获取文档失败 {chunkId}: {e}")
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
        try:
            return self.es_conn.createIdx(indexName, schema)
        except Exception as e:
            logging.error(f"创建索引失败: {e}")
            return False

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        """删除索引"""
        try:
            return self.es_conn.deleteIdx(indexName)
        except Exception as e:
            logging.error(f"删除索引失败: {e}")
            return False

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        """插入文档"""
        try:
            return self.es_conn.insert(documents, indexName)
        except Exception as e:
            logging.error(f"插入文档失败: {e}")
            return []

    def update(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> int:
        """更新文档"""
        try:
            return self.es_conn.update(documents, indexName)
        except Exception as e:
            logging.error(f"更新文档失败: {e}")
            return 0

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        """删除文档"""
        try:
            return self.es_conn.delete(condition, indexName)
        except Exception as e:
            logging.error(f"删除文档失败: {e}")
            return 0

    def sql(self, sql: str, fetch_size: int, format: str):
        """执行SQL查询"""
        try:
            return self.es_conn.sql(sql, fetch_size, format)
        except Exception as e:
            logging.error(f"SQL查询失败: {e}")
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
        # 构建bool查询
        bqry = {
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
                    bqry["bool"]["filter"].append({"terms": {"doc_id": v}})
                elif k == "available_int":
                    bqry["bool"]["filter"].append({"term": {"available_int": v}})
                else:
                    bqry["bool"]["filter"].append({"term": {k: v}})

        # 构建查询
        s = {"query": bqry}

        # 添加高亮
        if highlightFields:
            s["highlight"] = {"fields": {}}
            for field in highlightFields:
                s["highlight"]["fields"][field] = {}

        # 添加源字段
        if selectFields:
            s["_source"] = selectFields

        # 只处理文本搜索表达式
        for m in matchExprs:
            if isinstance(m, MatchTextExpr):
                minimum_should_match = m.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    minimum_should_match = str(int(minimum_should_match * 100)) + "%"

                bqry["bool"]["must"].append({
                    "query_string": {
                        "fields": m.fields,
                        "type": "best_fields",
                        "query": m.matching_text,
                        "minimum_should_match": minimum_should_match,
                        "boost": 1
                    }
                })

        # 添加排序
        if orderBy and orderBy.fields:
            s["sort"] = []
            for field, order in orderBy.fields:
                order_str = "asc" if order == 0 else "desc"
                s["sort"].append({field: {"order": order_str}})

        # 添加聚合
        if aggFields:
            s["aggs"] = {}
            for fld in aggFields:
                s["aggs"][f'aggs_{fld}'] = {
                    "terms": {
                        "field": fld,
                        "size": 1000000
                    }
                }

        # 添加分页
        if limit > 0:
            s["from"] = offset
            s["size"] = limit

        return s

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
