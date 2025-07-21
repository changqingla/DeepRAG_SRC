# -*- coding: utf-8 -*-
"""
DeepRAG 嵌入工具模块

本模块为文档嵌入提供实用工具函数和辅助类，包括：
- 配置管理和模型选择
- 嵌入结果分析和质量评估
- 相似度计算和聚类分析
- 多格式数据导出功能

主要组件：
1. EmbeddingResult: 嵌入结果数据类
2. EmbeddingConfigManager: 配置管理器
3. EmbeddingAnalyzer: 结果分析器
4. EmbeddingExporter: 数据导出器

作者: HU TAO
许可证: Apache 2.0
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict

from .chunk_embedder import EmbeddingConfig


@dataclass
class EmbeddingResult:
    """
    嵌入操作结果数据类

    存储嵌入操作的完整结果，包括处理后的分块、统计信息和模型信息。
    """
    chunks: List[Dict[str, Any]]  # 嵌入后的分块列表
    token_count: int              # 处理的 token 总数
    vector_size: int              # 向量维度
    processing_time: float        # 处理耗时（秒）
    model_info: Dict[str, str]    # 模型信息

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 结果字典
        """
        return {
            "chunks": self.chunks,
            "token_count": self.token_count,
            "vector_size": self.vector_size,
            "processing_time": self.processing_time,
            "model_info": self.model_info
        }

    def save_to_file(self, file_path: Path):
        """
        保存结果到 JSON 文件

        Args:
            file_path (Path): 保存路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class EmbeddingConfigManager:
    """
    嵌入配置管理器

    管理各种嵌入模型的预定义配置，提供配置的创建、保存和加载功能。
    """

    @classmethod
    def create_custom_config(cls,
                           model_factory: str,
                           model_name: str,
                           api_key: str = "",
                           **kwargs) -> EmbeddingConfig:
        """
        创建自定义嵌入配置

        Args:
            model_factory (str): 模型工厂名称
            model_name (str): 模型名称
            api_key (str): API 密钥（如果需要）
            **kwargs: 其他参数

        Returns:
            EmbeddingConfig: 配置实例
        """
        return EmbeddingConfig(
            model_factory=model_factory,
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )


class EmbeddingAnalyzer:
    """
    嵌入结果分析器

    提供嵌入向量的质量分析、相似度计算和聚类功能。
    """

    @staticmethod
    def analyze_embeddings(chunks: List[Dict[str, Any]], vector_field: str = None) -> Dict[str, Any]:
        """
        分析嵌入结果

        计算向量的统计信息，包括向量范数、相似度分布等指标。

        Args:
            chunks (List[Dict[str, Any]]): 包含嵌入向量的分块列表
            vector_field (str): 向量字段名（如果为 None 则自动检测）

        Returns:
            Dict[str, Any]: 分析结果
        """
        if not chunks:
            return {"error": "没有分块可供分析"}

        # 如果未提供向量字段名，则自动检测
        if vector_field is None:
            for key in chunks[0].keys():
                if key.startswith("q_") and key.endswith("_vec"):
                    vector_field = key
                    break

            if vector_field is None:
                return {"error": "在分块中未找到向量字段"}

        # 提取向量
        vectors = []
        for chunk in chunks:
            if vector_field in chunk:
                vectors.append(np.array(chunk[vector_field]))

        if not vectors:
            return {"error": f"在字段 {vector_field} 中未找到向量"}
        
        vectors = np.array(vectors)

        # 计算统计信息
        vector_norms = np.linalg.norm(vectors, axis=1)

        # 计算成对相似度（对大数据集进行采样）
        sample_size = min(100, len(vectors))
        sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
        sample_vectors = vectors[sample_indices]

        # 余弦相似度矩阵
        similarities = np.dot(sample_vectors, sample_vectors.T) / (
            np.linalg.norm(sample_vectors, axis=1)[:, np.newaxis] *
            np.linalg.norm(sample_vectors, axis=1)[np.newaxis, :]
        )

        # 移除对角线（自相似度）
        similarities = similarities[~np.eye(similarities.shape[0], dtype=bool)]
        
        return {
            "total_chunks": len(chunks),
            "vector_dimension": len(vectors[0]),
            "vector_field": vector_field,
            "vector_norm_stats": {
                "mean": float(np.mean(vector_norms)),
                "std": float(np.std(vector_norms)),
                "min": float(np.min(vector_norms)),
                "max": float(np.max(vector_norms))
            },
            "similarity_stats": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            },
            "sample_size": sample_size
        }
    
    @staticmethod
    def find_similar_chunks(chunks: List[Dict[str, Any]],
                          query_chunk: Dict[str, Any],
                          top_k: int = 5,
                          vector_field: str = None) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        查找与查询分块相似的分块

        使用余弦相似度计算查询分块与其他分块的相似度，返回最相似的前 k 个结果。

        Args:
            chunks (List[Dict[str, Any]]): 包含嵌入向量的分块列表
            query_chunk (Dict[str, Any]): 查询分块（包含嵌入向量）
            top_k (int): 返回的最相似分块数量
            vector_field (str): 向量字段名（如果为 None 则自动检测）

        Returns:
            List[Tuple[int, float, Dict[str, Any]]]: (索引, 相似度, 分块) 元组列表
        """
        if not chunks or not query_chunk:
            return []
        
        # Auto-detect vector field
        if vector_field is None:
            for key in chunks[0].keys():
                if key.startswith("q_") and key.endswith("_vec"):
                    vector_field = key
                    break
        
        if vector_field not in query_chunk:
            return []
        
        query_vector = np.array(query_chunk[vector_field])
        similarities = []
        
        for i, chunk in enumerate(chunks):
            if vector_field in chunk:
                chunk_vector = np.array(chunk[vector_field])
                
                # Cosine similarity
                similarity = np.dot(query_vector, chunk_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
                )
                
                similarities.append((i, float(similarity), chunk))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @staticmethod
    def cluster_chunks(chunks: List[Dict[str, Any]],
                      n_clusters: int = 5,
                      vector_field: str = None) -> Dict[str, Any]:
        """
        基于嵌入向量对分块进行聚类

        使用 K-means 算法对分块进行聚类，并计算聚类质量指标。

        Args:
            chunks (List[Dict[str, Any]]): 包含嵌入向量的分块列表
            n_clusters (int): 聚类数量
            vector_field (str): 向量字段名（如果为 None 则自动检测）

        Returns:
            Dict[str, Any]: 聚类结果，包含聚类标签、质量指标等
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            return {"error": "scikit-learn not available for clustering"}
        
        if not chunks:
            return {"error": "No chunks to cluster"}
        
        # Auto-detect vector field
        if vector_field is None:
            for key in chunks[0].keys():
                if key.startswith("q_") and key.endswith("_vec"):
                    vector_field = key
                    break
        
        # Extract vectors
        vectors = []
        for chunk in chunks:
            if vector_field in chunk:
                vectors.append(chunk[vector_field])
        
        if not vectors:
            return {"error": f"No vectors found in field {vector_field}"}
        
        vectors = np.array(vectors)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(vectors, cluster_labels)
        
        # Group chunks by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "index": i,
                "chunk": chunks[i]
            })
        
        return {
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette_avg),
            "cluster_sizes": {str(k): len(v) for k, v in clusters.items()},
            "clusters": {str(k): v for k, v in clusters.items()}
        }


class EmbeddingExporter:
    """
    嵌入结果导出器

    提供多种格式的嵌入向量和元数据导出功能。
    """

    @staticmethod
    def export_vectors_only(chunks: List[Dict[str, Any]],
                           output_path: Path,
                           vector_field: str = None,
                           format: str = "npy"):
        """
        仅导出分块中的向量数据

        Args:
            chunks (List[Dict[str, Any]]): 包含嵌入向量的分块列表
            output_path (Path): 输出文件路径
            vector_field (str): 向量字段名（如果为 None 则自动检测）
            format (str): 导出格式（"npy", "txt", "csv"）
        """
        if not chunks:
            raise ValueError("没有分块可供导出")

        # 自动检测向量字段
        if vector_field is None:
            for key in chunks[0].keys():
                if key.startswith("q_") and key.endswith("_vec"):
                    vector_field = key
                    break

        # 提取向量
        vectors = []
        for chunk in chunks:
            if vector_field in chunk:
                vectors.append(chunk[vector_field])

        if not vectors:
            raise ValueError(f"在字段 {vector_field} 中未找到向量")

        vectors = np.array(vectors)

        if format == "npy":
            np.save(output_path, vectors)
        elif format == "txt":
            np.savetxt(output_path, vectors)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame(vectors)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logging.info(f"已导出 {len(vectors)} 个向量到 {output_path}")
    
    @staticmethod
    def export_with_metadata(chunks: List[Dict[str, Any]],
                           output_path: Path,
                           include_vectors: bool = True):
        """
        导出包含元数据的分块信息

        Args:
            chunks (List[Dict[str, Any]]): 包含嵌入向量的分块列表
            output_path (Path): 输出文件路径
            include_vectors (bool): 是否包含向量数据
        """
        export_data = []
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "index": i,
                "content": chunk.get("content_with_weight", ""),
                "doc_name": chunk.get("docnm_kwd", ""),
                "content_length": len(chunk.get("content_with_weight", "")),
            }
            
            # 添加向量信息
            for key in chunk.keys():
                if key.startswith("q_") and key.endswith("_vec"):
                    chunk_data["vector_field"] = key
                    chunk_data["vector_dimension"] = len(chunk[key])
                    if include_vectors:
                        chunk_data["vector"] = chunk[key]
                    break

            export_data.append(chunk_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logging.info(f"已导出 {len(chunks)} 个包含元数据的分块到 {output_path}")
