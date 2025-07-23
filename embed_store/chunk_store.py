#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的文档分块存储模块

专注于将解析后的文档分块存储到Elasticsearch

作者: Hu Tao
许可证: Apache 2.0
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from es_connection import SimpleESConnection

logger = logging.getLogger('embed_store.chunk_store')


class DocumentStore:
    """
    简化的文档存储器
    专注于将解析后的文档分块存储到ES
    """

    def __init__(self,
                 es_host: str = "http://localhost:9200",
                 index_name: str = "documents",
                 **es_kwargs):
        """
        初始化文档存储器

        Args:
            es_host: Elasticsearch地址
            index_name: 索引名称
            **es_kwargs: ES连接参数
        """
        self.index_name = index_name
        self.es_conn = SimpleESConnection(es_host, **es_kwargs)
        self.vector_dim = None

    def _detect_vector_dimension(self, chunks: List[Dict[str, Any]]) -> int:
        """
        从分块数据中检测向量维度

        Args:
            chunks: 分块数据列表

        Returns:
            int: 向量维度
        """
        for chunk in chunks:
            for key, value in chunk.items():
                if key.startswith("q_") and key.endswith("_vec"):
                    if isinstance(value, list) and len(value) > 0:
                        return len(value)

        raise ValueError("未找到向量字段或向量为空")

    def _normalize_chunk(self, chunk: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        标准化分块数据格式 - 基于实际数据结构，保留所有原始字段并添加必要的DeepRAG兼容字段

        Args:
            chunk: 原始分块数据
            chunk_index: 分块索引

        Returns:
            Dict: 标准化后的分块数据
        """
        # 生成唯一ID
        chunk_id = str(uuid.uuid4())

        # 第一步：完全保留所有原始字段（不做任何修改）
        normalized = chunk.copy()

        # 第二步：添加必要的标识字段
        normalized["id"] = chunk_id
        normalized["doc_id"] = chunk.get("docnm_kwd", "unknown")  # 使用文档名作为doc_id
        normalized["chunk_index"] = chunk_index

        # 第三步：添加DeepRAG系统需要但数据中不存在的字段（使用默认值）
        # 这些字段在检索中可能被用到，提供默认值确保兼容性

        # 重要字段（如果不存在则设为空）
        if "important_kwd" not in normalized:
            normalized["important_kwd"] = []
        if "important_tks" not in normalized:
            normalized["important_tks"] = ""
        if "question_tks" not in normalized:
            normalized["question_tks"] = ""
        if "question_kwd" not in normalized:
            normalized["question_kwd"] = []

        # 状态字段
        if "available_int" not in normalized:
            normalized["available_int"] = 1  # 默认可用

        # 时间字段
        if "create_timestamp_flt" not in normalized:
            normalized["create_timestamp_flt"] = datetime.now().timestamp()
        if "create_time" not in normalized:
            normalized["create_time"] = datetime.now().isoformat()

        # 其他可选字段
        if "img_id" not in normalized:
            normalized["img_id"] = ""
        if "knowledge_graph_kwd" not in normalized:
            normalized["knowledge_graph_kwd"] = []

        # 第四步：确保数据类型正确
        # 确保关键词字段是列表格式
        for field in ["important_kwd", "question_kwd", "knowledge_graph_kwd"]:
            if isinstance(normalized.get(field), str):
                normalized[field] = [normalized[field]] if normalized[field] else []

        return normalized

    def create_index(self, vector_dim: int = None) -> bool:
        """
        创建索引

        Args:
            vector_dim: 向量维度，如果不提供则使用检测到的维度

        Returns:
            bool: 创建是否成功
        """
        if vector_dim is None:
            vector_dim = self.vector_dim or 1024

        return self.es_conn.create_index(self.index_name, vector_dim)

    def store_chunks(self,
                    chunks: List[Dict[str, Any]],
                    batch_size: int = 100,
                    progress_callback: Optional[callable] = None) -> Tuple[int, List[str]]:
        """
        存储文档分块

        Args:
            chunks: 分块数据列表
            batch_size: 批量大小
            progress_callback: 进度回调函数

        Returns:
            Tuple[int, List[str]]: (成功数量, 错误列表)
        """
        if not chunks:
            raise ValueError("没有要存储的分块数据")

        # 检测向量维度
        self.vector_dim = self._detect_vector_dimension(chunks)
        logger.info(f"检测到向量维度: {self.vector_dim}")

        # 创建索引
        if not self.create_index(self.vector_dim):
            raise Exception(f"创建索引失败: {self.index_name}")

        # 标准化分块数据
        if progress_callback:
            progress_callback(0.1, "正在标准化分块数据...")

        normalized_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                normalized = self._normalize_chunk(chunk, i)
                normalized_chunks.append(normalized)
            except Exception as e:
                logger.error(f"标准化分块 {i} 失败: {e}")
                continue

        logger.info(f"标准化完成: {len(normalized_chunks)}/{len(chunks)} 个分块")

        # 批量存储
        if progress_callback:
            progress_callback(0.2, "开始批量存储...")

        total_success = 0
        all_errors = []

        for i in range(0, len(normalized_chunks), batch_size):
            batch = normalized_chunks[i:i + batch_size]

            try:
                result = self.es_conn.bulk_index(self.index_name, batch)
                total_success += result["success"]
                all_errors.extend(result["errors"])

                # 进度回调
                if progress_callback:
                    progress = 0.2 + 0.7 * (i + len(batch)) / len(normalized_chunks)
                    progress_callback(progress, f"已存储 {total_success} 个分块")

            except Exception as e:
                error_msg = f"批次 {i//batch_size + 1} 存储失败: {e}"
                logger.error(error_msg)
                all_errors.append(error_msg)

        if progress_callback:
            progress_callback(1.0, f"存储完成: {total_success} 个分块")

        logger.info(f"存储完成: 成功 {total_success} 个，错误 {len(all_errors)} 个")
        return total_success, all_errors

    def load_and_store_from_file(self,
                                file_path: str,
                                batch_size: int = 100,
                                progress_callback: Optional[callable] = None) -> Tuple[int, List[str]]:
        """
        从JSON文件加载并存储分块数据

        Args:
            file_path: JSON文件路径
            batch_size: 批量大小
            progress_callback: 进度回调函数

        Returns:
            Tuple[int, List[str]]: (成功数量, 错误列表)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"从文件加载分块数据: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            if not isinstance(chunks, list):
                raise ValueError("文件内容必须是分块数组")

            logger.info(f"加载了 {len(chunks)} 个分块")
            return self.store_chunks(chunks, batch_size, progress_callback)

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON文件格式错误: {e}")

    def search_documents(self,
                        query_text: str,
                        size: int = 10,
                        doc_name: str = None) -> List[Dict[str, Any]]:
        """
        搜索文档

        Args:
            query_text: 查询文本
            size: 返回结果数量
            doc_name: 文档名称过滤

        Returns:
            List[Dict]: 搜索结果
        """
        # 构建查询
        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title^2", "content", "content_tokens"]
                        }
                    }
                ]
            }
        }

        # 添加文档名称过滤
        if doc_name:
            query["bool"]["filter"] = [
                {"term": {"doc_name": doc_name}}
            ]

        try:
            response = self.es_conn.search(self.index_name, query, size)
            hits = response.get("hits", {}).get("hits", [])

            results = []
            for hit in hits:
                result = hit["_source"]
                result["_score"] = hit["_score"]
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """
        获取文档统计信息

        Returns:
            Dict: 统计信息
        """
        try:
            # 直接使用ES客户端进行聚合查询
            # 因为SimpleESConnection.search()会包装查询，不适合聚合查询
            stats_query = {
                "size": 0,
                "aggs": {
                    "doc_count": {
                        "cardinality": {
                            "field": "doc_name"
                        }
                    },
                    "docs_by_name": {
                        "terms": {
                            "field": "doc_name",
                            "size": 100
                        }
                    }
                }
            }

            # 直接使用ES客户端，避免SimpleESConnection的查询包装
            response = self.es_conn.es.search(
                index=self.index_name,
                body=stats_query
            )

            total_chunks = response.get("hits", {}).get("total", {}).get("value", 0)
            unique_docs = response.get("aggregations", {}).get("doc_count", {}).get("value", 0)
            docs_detail = response.get("aggregations", {}).get("docs_by_name", {}).get("buckets", [])

            return {
                "index_name": self.index_name,
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
                "vector_dimension": self.vector_dim,
                "documents": [
                    {"name": doc["key"], "chunk_count": doc["doc_count"]}
                    for doc in docs_detail
                ]
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "index_name": self.index_name,
                "error": str(e)
            }

    def delete_index(self) -> bool:
        """删除索引"""
        return self.es_conn.delete_index(self.index_name)

    def index_exists(self) -> bool:
        """检查索引是否存在"""
        return self.es_conn.index_exists(self.index_name)

    def get_health(self) -> Dict[str, Any]:
        """获取ES健康状态"""
        return self.es_conn.get_health()


def simple_progress_callback(progress: float, message: str):
    """简单的进度回调函数"""
    if progress >= 0:
        print(f"进度: {progress:.1%} - {message}")
    else:
        print(f"错误: {message}")


# 使用示例
if __name__ == "__main__":
    import sys

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 创建文档存储器
    store = DocumentStore(
        es_host="http://10.0.100.36:9201",
        index_name="test_documents"
    )

    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python chunk_store.py <json_file_path>")
        print("示例: python chunk_store.py markdown_chunks_embedded.json")
        sys.exit(1)

    json_file = sys.argv[1]

    try:
        # 存储文档
        print(f"开始存储文档: {json_file}")
        success_count, errors = store.load_and_store_from_file(
            json_file,
            batch_size=50,
            progress_callback=simple_progress_callback
        )

        print(f"\n存储结果:")
        print(f"成功: {success_count} 个分块")
        print(f"错误: {len(errors)} 个")

        if errors:
            print("\n错误详情:")
            for error in errors[:5]:  # 只显示前5个错误
                print(f"  - {error}")

        # 显示统计信息
        stats = store.get_document_stats()
        print(f"\n索引统计:")
        print(f"索引名称: {stats['index_name']}")
        print(f"总分块数: {stats.get('total_chunks', 0)}")
        print(f"文档数量: {stats.get('unique_documents', 0)}")
        print(f"向量维度: {stats.get('vector_dimension', 'N/A')}")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)