#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的Elasticsearch连接模块

本模块提供基于RAGFlow原有逻辑的独立Elasticsearch连接实现，
不依赖外部模块，完全复用RAGFlow的存储算法。

作者: RAGFlow开发团队
许可证: Apache 2.0
"""

import logging
import re
import json
import time
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch_dsl import Index
from elastic_transport import ConnectionTimeout

# RAGFlow中的常量配置
ATTEMPT_TIME = 2  # 重试次数

logger = logging.getLogger('embed_store.es_connection')


class IndependentESConnection:
    """
    基于RAGFlow ESConnection逻辑的独立Elasticsearch连接类

    本类仅实现分块存储所需的核心方法，不依赖RAGFlow的settings或其他模块。
    完全复用RAGFlow的存储算法逻辑，包括索引创建、批量插入、错误处理等。
    """

    def __init__(self, es_config: Dict[str, Any] = None):
        """
        初始化Elasticsearch连接

        Args:
            es_config: Elasticsearch配置字典，包含hosts、用户名、密码等信息
        """
        # 默认ES配置（基于RAGFlow的settings配置）
        self.default_config = {
            "hosts": "http://localhost:9200",  # ES服务器地址
            "username": "",                    # 用户名（可选）
            "password": "",                    # 密码（可选）
            "timeout": 600                     # 连接超时时间（秒）
        }

        # 使用提供的配置或默认配置
        if es_config:
            self.es_config = {**self.default_config, **es_config}
        else:
            # 尝试使用RAGFlow的配置（如果可用）
            try:
                from rag import settings
                self.es_config = settings.ES
                logger.info("使用RAGFlow的ES配置")
            except ImportError:
                self.es_config = self.default_config
                logger.info("使用默认ES配置")

        # 初始化Elasticsearch客户端
        self._init_elasticsearch()

        # 加载mapping配置
        self._load_mapping()
    
    def _init_elasticsearch(self):
        """初始化Elasticsearch客户端（基于RAGFlow的逻辑）"""
        logger.info(f"正在连接Elasticsearch: {self.es_config['hosts']}")

        # 按照RAGFlow的重试逻辑进行连接
        for attempt in range(ATTEMPT_TIME):
            try:
                # 解析主机地址
                hosts = self.es_config["hosts"]
                if isinstance(hosts, str):
                    hosts = hosts.split(",")  # 支持多个主机地址

                # 设置身份验证
                auth = None
                if self.es_config.get("username") and self.es_config.get("password"):
                    auth = (self.es_config["username"], self.es_config["password"])

                # 创建Elasticsearch客户端
                self.es = Elasticsearch(
                    hosts,
                    basic_auth=auth,
                    verify_certs=False,  # 不验证SSL证书
                    timeout=self.es_config.get("timeout", 600)
                )

                # 测试连接
                health = self.es.cluster.health()
                logger.info(f"Elasticsearch连接成功，集群状态: {health['status']}")
                return

            except Exception as e:
                logger.warning(f"Elasticsearch连接尝试 {attempt + 1} 失败: {e}")
                if attempt == ATTEMPT_TIME - 1:
                    raise Exception(f"经过 {ATTEMPT_TIME} 次尝试后仍无法连接到Elasticsearch: {e}")
                time.sleep(2)  # 重试前等待2秒
    
    def _load_mapping(self):
        """加载mapping配置（基于RAGFlow的逻辑）"""
        try:
            # 尝试加载RAGFlow的mapping文件
            from api.utils.file_utils import get_project_base_directory
            mapping_file = Path(get_project_base_directory()) / "conf" / "mapping.json"

            if mapping_file.exists():
                with open(mapping_file, "r") as f:
                    self.mapping = json.load(f)
                logger.info("已加载RAGFlow的mapping配置")
                return
        except ImportError:
            pass

        # 如果RAGFlow的mapping不可用，使用默认mapping
        self.mapping = self._get_default_mapping()
        logger.info("使用默认mapping配置")
    
    def _get_default_mapping(self) -> Dict[str, Any]:
        """获取默认mapping配置（基于RAGFlow的mapping.json）"""
        return {
            "settings": {
                "index": {
                    "number_of_shards": 1,              # 分片数量
                    "number_of_replicas": 0,            # 副本数量
                    "max_result_window": 2000000,       # 最大搜索结果窗口
                    "highlight.max_analyzed_offset": 2000000  # 高亮分析偏移量
                },
                "analysis": {
                    "analyzer": {
                        "my_ik": {                      # 中文分词器
                            "tokenizer": "ik_max_word",
                            "filter": ["lowercase"]
                        },
                        "my_ws": {                      # 空格分词器
                            "tokenizer": "whitespace",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "dynamic_templates": [
                    {
                        "dense_vector": {                # 动态向量字段模板
                            "match": "*_vec",            # 匹配所有以_vec结尾的字段
                            "mapping": {
                                "type": "dense_vector",   # 向量类型
                                "index": True,           # 启用索引
                                "similarity": "cosine"   # 余弦相似度
                            }
                        }
                    }
                ],
                "properties": {
                    "id": {"type": "keyword"},                    # 分块唯一ID
                    "doc_id": {"type": "keyword"},               # 文档ID
                    "kb_id": {"type": "keyword"},                # 知识库ID
                    "content_with_weight": {                     # 分块内容
                        "type": "text",
                        "analyzer": "my_ik",                     # 使用中文分词
                        "search_analyzer": "my_ik"
                    },
                    "content_ltks": {                            # 基础分词结果
                        "type": "text",
                        "analyzer": "my_ws",                     # 使用空格分词
                        "search_analyzer": "my_ws"
                    },
                    "content_sm_ltks": {                         # 细粒度分词结果
                        "type": "text",
                        "analyzer": "my_ws",
                        "search_analyzer": "my_ws"
                    },
                    "docnm_kwd": {"type": "keyword"},           # 文档名称
                    "title_tks": {                              # 标题分词
                        "type": "text",
                        "analyzer": "my_ik",
                        "search_analyzer": "my_ik"
                    },
                    "title_sm_tks": {                           # 标题细粒度分词
                        "type": "text",
                        "analyzer": "my_ws",
                        "search_analyzer": "my_ws"
                    },
                    "page_num_int": {"type": "integer"},        # 页码
                    "position_int": {"type": "integer"},        # 位置坐标
                    "top_int": {"type": "integer"},             # 顶部位置
                    "available_int": {"type": "integer"},       # 可用状态
                    "create_time": {"type": "date"},            # 创建时间
                    "create_timestamp_flt": {"type": "float"}   # 创建时间戳
                }
            }
        }
    
    def createIdx(self, index_name: str, kb_id: str, vector_size: int) -> bool:
        """
        创建索引（基于RAGFlow的createIdx逻辑）

        Args:
            index_name: 索引名称
            kb_id: 知识库ID（在ES中不使用，保留用于兼容性）
            vector_size: 向量维度（用于更新mapping配置）

        Returns:
            True表示创建成功
        """
        # 如果索引已存在，直接返回成功
        if self.indexExist(index_name, kb_id):
            return True

        try:
            # 为特定向量维度更新mapping配置
            mapping = copy.deepcopy(self.mapping)

            # 添加特定向量字段的mapping
            vector_field = f"q_{vector_size}_vec"
            mapping["mappings"]["properties"][vector_field] = {
                "type": "dense_vector",      # 向量类型
                "index": True,               # 启用索引
                "similarity": "cosine",      # 余弦相似度
                "dims": vector_size          # 向量维度
            }

            # 创建索引
            from elasticsearch.client import IndicesClient
            result = IndicesClient(self.es).create(
                index=index_name,
                settings=mapping["settings"],
                mappings=mapping["mappings"]
            )

            logger.info(f"成功创建索引: {index_name}，向量维度: {vector_size}")
            return True

        except Exception as e:
            logger.error(f"创建索引 {index_name} 失败: {e}")
            return False
    
    def deleteIdx(self, index_name: str, kb_id: str = ""):
        """
        删除索引（基于RAGFlow的deleteIdx逻辑）

        Args:
            index_name: 索引名称
            kb_id: 知识库ID（在ES中不使用）
        """
        if len(kb_id) > 0:
            # 在RAGFlow中，索引在多个知识库间共享，所以不删除
            logger.info(f"跳过索引删除 {index_name}（提供了kb_id）")
            return

        try:
            self.es.indices.delete(index=index_name, allow_no_indices=True)
            logger.info(f"已删除索引: {index_name}")
        except NotFoundError:
            logger.info(f"索引 {index_name} 不存在，无需删除")
        except Exception as e:
            logger.error(f"删除索引 {index_name} 失败: {e}")
    
    def indexExist(self, index_name: str, kb_id: str = "") -> bool:
        """
        检查索引是否存在（基于RAGFlow的indexExist逻辑）

        Args:
            index_name: 索引名称
            kb_id: 知识库ID（在ES中不使用）

        Returns:
            True表示索引存在
        """
        s = Index(index_name, self.es)

        # 按照RAGFlow的重试逻辑检查索引存在性
        for attempt in range(ATTEMPT_TIME):
            try:
                return s.exists()
            except Exception as e:
                logger.warning(f"索引存在性检查尝试 {attempt + 1} 失败: {e}")
                # 如果是超时或冲突错误，继续重试
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue
                break

        return False
    
    def insert(self, documents: List[Dict[str, Any]], index_name: str, kb_id: str = "") -> List[str]:
        """
        插入文档（基于RAGFlow的insert逻辑）

        Args:
            documents: 要插入的文档列表
            index_name: 索引名称
            kb_id: 知识库ID（在ES中不使用）

        Returns:
            错误消息列表（如果成功则为空列表）
        """
        # 准备批量操作（与RAGFlow相同的逻辑）
        operations = []
        for doc in documents:
            # 确保文档格式正确
            assert "_id" not in doc, "文档不应包含_id字段"
            assert "id" in doc, "文档必须包含id字段"

            doc_copy = copy.deepcopy(doc)
            meta_id = doc_copy.pop("id", "")  # 提取文档ID作为ES的_id

            # 添加索引操作和文档数据
            operations.append({"index": {"_index": index_name, "_id": meta_id}})
            operations.append(doc_copy)

        # 执行批量插入（按照RAGFlow的重试逻辑）
        for attempt in range(ATTEMPT_TIME):
            try:
                result = self.es.bulk(
                    index=index_name,
                    operations=operations,
                    refresh=False,           # 不立即刷新索引
                    timeout="60s"           # 60秒超时
                )

                # 检查错误（与RAGFlow相同的逻辑）
                if re.search(r"False", str(result["errors"]), re.IGNORECASE):
                    return []  # 没有错误

                # 收集错误消息
                errors = []
                for item in result["items"]:
                    for action in ["create", "delete", "index", "update"]:
                        if action in item and "error" in item[action]:
                            error_msg = f"{item[action]['_id']}:{item[action]['error']}"
                            errors.append(error_msg)

                return errors

            except Exception as e:
                logger.warning(f"批量插入尝试 {attempt + 1} 失败: {e}")
                # 如果是超时错误，等待后重试
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                return [str(e)]

        return ["经过多次尝试后批量插入仍然失败"]
    
    def health(self) -> Dict[str, Any]:
        """获取Elasticsearch健康状态"""
        try:
            health_dict = dict(self.es.cluster.health())
            health_dict["type"] = "elasticsearch"
            return health_dict
        except Exception as e:
            return {"type": "elasticsearch", "status": "error", "error": str(e)}
