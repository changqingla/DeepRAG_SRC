#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deeprag分块存储模块

本模块提供将已向量化的分块存储到Elasticsearch的功能，

作者: Hu Tao
许可证: Apache 2.0
"""

import os
import sys
import json
import logging
import copy
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import xxhash
from timeit import default_timer as timer

# 添加deeprag根目录到路径
current_dir = Path(__file__).parent.absolute()
deeprag_root = current_dir.parent
sys.path.insert(0, str(deeprag_root))

# 导入deeprag组件
from rag.nlp import rag_tokenizer
from rag.utils import num_tokens_from_string

# 导入我们的独立ES连接类
from es_connection import IndependentESConnection


class SimpleStoreConfig:
    """简化的存储配置类（无租户系统）"""

    def __init__(self,
                 index_name: str = "deeprag_vectors",
                 es_config: Dict[str, Any] = None,
                 batch_size: int = 8,
                 auto_create_index: bool = True):
        """
        初始化简化存储配置

        Args:
            index_name: Elasticsearch索引名称（默认: deeprag_vectors）
            es_config: Elasticsearch配置字典（默认连接localhost:9200）
            batch_size: 批量操作的批次大小
            auto_create_index: 是否自动创建索引
        """
        self.index_name = index_name
        self.es_config = es_config or {
            "hosts": "http://localhost:9200",
            "timeout": 600
        }
        self.batch_size = batch_size
        self.auto_create_index = auto_create_index

        # 简化：不需要复杂的ID层级
        self.doc_id = str(uuid.uuid4())  # 简单的文档ID

    @classmethod
    def create_simple(cls, index_name: str = None, **kwargs) -> 'SimpleStoreConfig':
        """创建简单配置"""
        if index_name:
            return cls(index_name=index_name, **kwargs)
        return cls(**kwargs)


class SimpleVectorStore:
    """
    简化的向量存储器（基于deeprag算法，无租户系统）

    本类提供将已向量化的分块直接存储到Elasticsearch的功能，
    使用deeprag原有的存储算法，但简化了层级结构。
    """

    def __init__(self, config: SimpleStoreConfig):
        """
        初始化简化向量存储器

        Args:
            config: 简化存储配置对象
        """
        self.config = config
        self.es_conn = IndependentESConnection(config.es_config)
        self.vector_size = None  # 向量维度，从数据中自动检测

        logging.info(f"简化向量存储器已初始化，索引: {config.index_name}")
        logging.info(f"ES连接: {config.es_config.get('hosts', 'localhost:9200')}")

    def _progress_callback(self, progress: float = None, msg: str = ""):
        """进度回调函数"""
        if progress is not None:
            logging.info(f"存储进度: {progress:.1%} - {msg}")
        else:
            logging.info(f"存储状态: {msg}")

    def _prepare_chunk_for_storage(self, chunk: Dict[str, Any], chunk_index: int = 0) -> Dict[str, Any]:
        """
        简化的分块数据准备（基于deeprag逻辑）

        Args:
            chunk: 原始分块数据
            chunk_index: 分块在批次中的索引

        Returns:
            准备好的ES存储分块数据
        """
        # 创建副本以避免修改原始数据
        prepared_chunk = copy.deepcopy(chunk)

        # 使用deeprag逻辑生成唯一ID（简化版）
        content = chunk.get("content_with_weight", "")
        chunk_id = xxhash.xxh64((content + str(chunk_index) + str(datetime.now().timestamp())).encode("utf-8")).hexdigest()

        # 简化的必需字段（保持deeprag兼容性）
        prepared_chunk["id"] = chunk_id
        prepared_chunk["doc_id"] = self.config.doc_id
        prepared_chunk["create_time"] = str(datetime.now()).replace("T", " ")[:19]
        prepared_chunk["create_timestamp_flt"] = datetime.now().timestamp()

        # 确保内容分词（与deeprag相同）
        if "content_ltks" not in prepared_chunk and content:
            prepared_chunk["content_ltks"] = rag_tokenizer.tokenize(content)

        if "content_sm_ltks" not in prepared_chunk and "content_ltks" in prepared_chunk:
            prepared_chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(prepared_chunk["content_ltks"])

        # 简化的默认值（保留核心字段）
        prepared_chunk.setdefault("docnm_kwd", prepared_chunk.get("docnm_kwd", "document"))
        prepared_chunk.setdefault("title_tks", "")
        prepared_chunk.setdefault("page_num_int", [1])
        prepared_chunk.setdefault("available_int", 1)

        return prepared_chunk

    def _detect_vector_size(self, chunks: List[Dict[str, Any]]) -> int:
        """从分块数据中检测向量维度"""
        for chunk in chunks:
            for key in chunk.keys():
                if key.startswith("q_") and key.endswith("_vec"):
                    vector = chunk[key]
                    if isinstance(vector, list) and len(vector) > 0:
                        return len(vector)

        raise ValueError("在分块数据中未找到向量字段")
    
    def store_vectors(self,
                     chunks: List[Dict[str, Any]],
                     callback=None) -> Tuple[int, List[str]]:
        """
        简化的向量存储方法（基于deeprag算法）

        Args:
            chunks: 要存储的已向量化分块列表
            callback: 进度回调函数

        Returns:
            元组：(成功存储数量, 错误消息列表)
        """
        if not chunks:
            raise ValueError("没有要存储的向量分块")

        if callback is None:
            callback = self._progress_callback

        callback(0.0, "开始向量存储...")

        # 检测向量维度
        if self.vector_size is None:
            self.vector_size = self._detect_vector_size(chunks)
            logging.info(f"检测到向量维度: {self.vector_size}")

        # 简化的索引创建（无kb_id层级）
        if self.config.auto_create_index:
            if not self.es_conn.indexExist(self.config.index_name, ""):
                callback(0.1, "正在创建Elasticsearch索引...")
                success = self.es_conn.createIdx(self.config.index_name, "", self.vector_size)
                if not success:
                    raise Exception(f"创建索引失败: {self.config.index_name}")
                logging.info(f"已创建索引: {self.config.index_name}")

        # 准备分块数据以供存储
        callback(0.2, "正在准备分块数据...")
        prepared_chunks = []
        for i, chunk in enumerate(chunks):
            prepared_chunk = self._prepare_chunk_for_storage(chunk, i)
            prepared_chunks.append(prepared_chunk)

        # 批量存储分块（与deeprag相同的逻辑）
        callback(0.3, "正在存储分块到Elasticsearch...")
        batch_size = self.config.batch_size
        error_messages = []
        stored_count = 0

        start_time = timer()

        for b in range(0, len(prepared_chunks), batch_size):
            batch_chunks = prepared_chunks[b:b + batch_size]

            # 使用deeprag逻辑插入批次（简化版）
            batch_errors = self.es_conn.insert(
                batch_chunks,
                self.config.index_name,
                ""  # 无kb_id层级
            )

            if batch_errors:
                error_messages.extend(batch_errors)
                logging.warning(f"批次 {b//batch_size + 1} 出现错误: {batch_errors}")
            else:
                stored_count += len(batch_chunks)

            # 进度回调
            progress = 0.3 + 0.6 * (b + batch_size) / len(prepared_chunks)
            callback(progress, f"已存储 {stored_count}/{len(prepared_chunks)} 个分块")

        processing_time = timer() - start_time

        if error_messages:
            callback(-1, f"存储完成但有错误: {len(error_messages)} 个失败")
            logging.error(f"存储错误: {error_messages}")
        else:
            callback(1.0, f"成功存储 {stored_count} 个分块")
            logging.info(f"成功存储 {stored_count} 个分块，耗时 {processing_time:.2f}秒")

        return stored_count, error_messages

    def get_index_info(self) -> Dict[str, Any]:
        """获取存储索引的信息"""
        return {
            "index_name": self.config.index_name,
            "doc_id": self.config.doc_id,
            "vector_size": self.vector_size,
            "index_exists": self.es_conn.indexExist(self.config.index_name, ""),
            "es_hosts": self.config.es_config.get("hosts", "localhost:9200")
        }

    def delete_index(self) -> bool:
        """删除存储索引"""
        try:
            self.es_conn.deleteIdx(self.config.index_name, "")  # 无kb_id
            logging.info(f"已删除索引: {self.config.index_name}")
            return True
        except Exception as e:
            logging.error(f"删除索引 {self.config.index_name} 失败: {e}")
            return False
