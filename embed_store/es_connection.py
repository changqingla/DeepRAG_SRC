#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的Elasticsearch连接模块

专注于ES连接和基本操作，去除复杂逻辑

作者: Hu Tao
许可证: Apache 2.0
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, NotFoundError

logger = logging.getLogger('embed_store.es_connection')


class SimpleESConnection:
    """
    简化的Elasticsearch连接类
    专注于基本的连接、索引创建和文档操作
    """

    def __init__(self, hosts: str = "http://localhost:9200", **kwargs):
        """
        初始化Elasticsearch连接

        Args:
            hosts: ES服务器地址
            **kwargs: 其他ES连接参数
        """
        self.hosts = hosts
        self.es = None
        self._connect(**kwargs)

    def _connect(self, **kwargs):
        """连接到Elasticsearch"""
        try:
            # 解析认证信息
            auth = None
            if 'username' in kwargs and 'password' in kwargs:
                auth = (kwargs['username'], kwargs['password'])

            # 创建ES客户端
            self.es = Elasticsearch(
                hosts=[self.hosts],
                basic_auth=auth,
                verify_certs=False,
                timeout=kwargs.get('timeout', 60)
            )

            # 测试连接
            health = self.es.cluster.health()
            logger.info(f"ES连接成功: {self.hosts}, 状态: {health['status']}")

        except Exception as e:
            logger.error(f"ES连接失败: {e}")
            raise

    def create_index(self, index_name: str, vector_dim: int = 1024) -> bool:
        """
        创建索引

        Args:
            index_name: 索引名称
            vector_dim: 向量维度

        Returns:
            bool: 创建是否成功
        """
        if self.index_exists(index_name):
            logger.info(f"索引 {index_name} 已存在")
            return True

        # DeepRAG兼容的mapping配置
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "text_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        },
                        "whitespace_analyzer": {
                            "tokenizer": "whitespace",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # === 基础标识字段 ===
                    "id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "docnm_kwd": {"type": "keyword"},

                    # === 内容字段（检索核心） ===
                    "content_with_weight": {
                        "type": "text",
                        "analyzer": "text_analyzer",
                        "store": True
                    },
                    "content_ltks": {
                        "type": "text",
                        "analyzer": "whitespace_analyzer",
                        "store": True
                    },
                    "content_sm_ltks": {
                        "type": "text",
                        "analyzer": "whitespace_analyzer",
                        "store": True
                    },

                    # === 标题字段（高权重检索） ===
                    "title_tks": {
                        "type": "text",
                        "analyzer": "whitespace_analyzer",
                        "store": True
                    },
                    "title_sm_tks": {
                        "type": "text",
                        "analyzer": "whitespace_analyzer",
                        "store": True
                    },

                    # === 重要字段（最高权重检索） ===
                    "important_kwd": {"type": "keyword"},
                    "important_tks": {
                        "type": "text",
                        "analyzer": "whitespace_analyzer",
                        "store": True
                    },
                    "question_tks": {
                        "type": "text",
                        "analyzer": "whitespace_analyzer",
                        "store": True
                    },
                    "question_kwd": {"type": "keyword"},

                    # === 位置和元数据字段 ===
                    "page_num_int": {"type": "integer"},
                    "position_int": {"type": "integer"},
                    "top_int": {"type": "integer"},

                    # === 状态和时间字段 ===
                    "available_int": {"type": "integer"},
                    "create_timestamp_flt": {"type": "float"},
                    "create_time": {"type": "date"},

                    # === 其他检索相关字段 ===
                    "img_id": {"type": "keyword"},
                    "knowledge_graph_kwd": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},

                    # === 向量字段（动态添加） ===
                    f"q_{vector_dim}_vec": {
                        "type": "dense_vector",
                        "dims": vector_dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        try:
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"成功创建索引: {index_name}")
            return True
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

    def index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        try:
            return self.es.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"检查索引存在性失败: {e}")
            return False

    def delete_index(self, index_name: str) -> bool:
        """删除索引"""
        try:
            if self.index_exists(index_name):
                self.es.indices.delete(index=index_name)
                logger.info(f"成功删除索引: {index_name}")
                return True
            else:
                logger.info(f"索引 {index_name} 不存在")
                return True
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            return False

    def bulk_index(self, index_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量索引文档

        Args:
            index_name: 索引名称
            documents: 文档列表

        Returns:
            Dict: 索引结果
        """
        if not documents:
            return {"success": 0, "errors": []}

        # 准备批量操作
        actions = []
        for doc in documents:
            action = {
                "_index": index_name,
                "_id": doc.get("id"),
                "_source": doc
            }
            actions.append(action)

        try:
            # 执行批量索引
            from elasticsearch.helpers import bulk
            success_count, errors = bulk(
                self.es,
                actions,
                index=index_name,
                chunk_size=100,
                request_timeout=60
            )

            logger.info(f"批量索引完成: 成功 {success_count} 个文档")
            return {
                "success": success_count,
                "errors": errors if isinstance(errors, list) else []
            }

        except Exception as e:
            logger.error(f"批量索引失败: {e}")
            return {"success": 0, "errors": [str(e)]}

    def search(self, index_name: str, query: Dict[str, Any], size: int = 10) -> Dict[str, Any]:
        """
        搜索文档

        Args:
            index_name: 索引名称
            query: 查询条件（可以是普通查询或包含KNN的完整查询体）
            size: 返回结果数量

        Returns:
            Dict: 搜索结果
        """
        try:
            # 检查是否为KNN查询或完整查询体
            if "knn" in query or "_source" in query or "highlight" in query:
                # 这是一个完整的查询体，直接使用
                if "size" not in query:
                    query["size"] = size
                body = query
            else:
                # 这是一个普通查询，需要包装
                body = {"query": query, "size": size}

            response = self.es.search(
                index=index_name,
                body=body
            )
            return response
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return {"hits": {"hits": []}}

    def get_health(self) -> Dict[str, Any]:
        """获取ES健康状态"""
        try:
            return self.es.cluster.health()
        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {"status": "error", "error": str(e)}