#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow存储工具模块

本模块提供用于将已向量化分块存储到Elasticsearch的
工具函数和辅助类。

作者: RAGFlow开发团队
许可证: Apache 2.0
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from chunk_store import SimpleStoreConfig


@dataclass
class StorageResult:
    """存储操作结果"""
    stored_count: int          # 成功存储的分块数量
    total_count: int           # 总分块数量
    error_count: int           # 错误数量
    error_messages: List[str]  # 错误消息列表
    processing_time: float     # 处理时间（秒）
    index_info: Dict[str, Any] # 索引信息

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def save_to_file(self, file_path: Path):
        """保存结果到JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_count == 0:
            return 0.0
        return self.stored_count / self.total_count

    @property
    def is_successful(self) -> bool:
        """检查存储是否成功"""
        return self.error_count == 0 and self.stored_count == self.total_count


class SimpleConfigManager:
    """简化的存储配置管理器"""

    @classmethod
    def create_simple_config(cls,
                           index_name: str = "ragflow_vectors",
                           **kwargs) -> SimpleStoreConfig:
        """
        创建简化存储配置

        Args:
            index_name: 索引名称
            **kwargs: 额外的配置参数

        Returns:
            SimpleStoreConfig实例
        """
        return SimpleStoreConfig.create_simple(index_name, **kwargs)

    @classmethod
    def create_with_es_host(cls,
                          es_host: str,
                          index_name: str = "ragflow_vectors",
                          **kwargs) -> SimpleStoreConfig:
        """
        创建带自定义ES地址的配置

        Args:
            es_host: ES服务器地址
            index_name: 索引名称
            **kwargs: 额外的配置参数

        Returns:
            SimpleStoreConfig实例
        """
        es_config = {"hosts": es_host, "timeout": 600}
        return SimpleStoreConfig(index_name=index_name, es_config=es_config, **kwargs)

    @classmethod
    def save_config(cls, config: SimpleStoreConfig, file_path: Path):
        """保存配置到文件"""
        config_dict = {
            "index_name": config.index_name,
            "es_config": config.es_config,
            "batch_size": config.batch_size,
            "auto_create_index": config.auto_create_index
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_config(cls, file_path: Path) -> SimpleStoreConfig:
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return SimpleStoreConfig(**config_dict)


class ChunkValidator:
    """Validator for chunk data before storage"""
    
    @staticmethod
    def validate_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate chunks before storage
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation result dictionary
        """
        if not chunks:
            return {
                "valid": False,
                "error": "No chunks provided",
                "warnings": []
            }
        
        warnings = []
        errors = []
        
        # Check for required fields
        required_fields = ["content_with_weight"]
        vector_fields = []
        
        for i, chunk in enumerate(chunks):
            # Check required fields
            for field in required_fields:
                if field not in chunk:
                    errors.append(f"Chunk {i}: Missing required field '{field}'")
            
            # Check for vector fields
            chunk_vector_fields = [k for k in chunk.keys() if k.startswith("q_") and k.endswith("_vec")]
            if not chunk_vector_fields:
                errors.append(f"Chunk {i}: No vector field found")
            else:
                vector_fields.extend(chunk_vector_fields)
            
            # Check content
            content = chunk.get("content_with_weight", "")
            if not content or not content.strip():
                warnings.append(f"Chunk {i}: Empty or whitespace-only content")
            
            # Check vector data
            for vf in chunk_vector_fields:
                vector = chunk[vf]
                if not isinstance(vector, list):
                    errors.append(f"Chunk {i}: Vector field '{vf}' is not a list")
                elif len(vector) == 0:
                    errors.append(f"Chunk {i}: Vector field '{vf}' is empty")
                elif not all(isinstance(v, (int, float)) for v in vector):
                    errors.append(f"Chunk {i}: Vector field '{vf}' contains non-numeric values")
        
        # Check vector consistency
        unique_vector_fields = list(set(vector_fields))
        if len(unique_vector_fields) > 1:
            warnings.append(f"Multiple vector field types found: {unique_vector_fields}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_chunks": len(chunks),
            "vector_fields": unique_vector_fields
        }
    
    @staticmethod
    def fix_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix common issues in chunks
        
        Args:
            chunks: List of chunks to fix
            
        Returns:
            Fixed chunks
        """
        fixed_chunks = []
        
        for chunk in chunks:
            fixed_chunk = chunk.copy()
            
            # Ensure content_with_weight exists
            if "content_with_weight" not in fixed_chunk:
                # Try to use other content fields
                content = (fixed_chunk.get("content", "") or 
                          fixed_chunk.get("text", "") or 
                          "Empty content")
                fixed_chunk["content_with_weight"] = content
            
            # Ensure docnm_kwd exists
            if "docnm_kwd" not in fixed_chunk:
                fixed_chunk["docnm_kwd"] = "unknown_document"
            
            # Ensure basic metadata exists
            fixed_chunk.setdefault("title_tks", "")
            fixed_chunk.setdefault("title_sm_tks", "")
            fixed_chunk.setdefault("page_num_int", [1])
            fixed_chunk.setdefault("position_int", [[1, 0, 0, 0, 0]])
            fixed_chunk.setdefault("top_int", [0])
            
            fixed_chunks.append(fixed_chunk)
        
        return fixed_chunks


class StorageAnalyzer:
    """Analyzer for storage operations"""
    
    @staticmethod
    def analyze_storage_result(result: StorageResult) -> Dict[str, Any]:
        """
        Analyze storage result
        
        Args:
            result: Storage result to analyze
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            "success_rate": result.success_rate,
            "is_successful": result.is_successful,
            "performance": {
                "chunks_per_second": result.stored_count / max(result.processing_time, 0.001),
                "processing_time": result.processing_time,
                "average_time_per_chunk": result.processing_time / max(result.stored_count, 1)
            },
            "errors": {
                "error_count": result.error_count,
                "error_rate": result.error_count / max(result.total_count, 1),
                "error_messages": result.error_messages[:10]  # Show first 10 errors
            },
            "index_info": result.index_info
        }
        
        # Add recommendations
        recommendations = []
        
        if result.success_rate < 1.0:
            recommendations.append("Some chunks failed to store. Check error messages for details.")
        
        if analysis["performance"]["chunks_per_second"] < 10:
            recommendations.append("Storage performance is slow. Consider increasing batch size.")
        
        if result.error_count > 0:
            recommendations.append("Storage had errors. Check Elasticsearch logs and connection.")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    @staticmethod
    def create_storage_report(result: StorageResult) -> str:
        """
        Create a human-readable storage report
        
        Args:
            result: Storage result
            
        Returns:
            Formatted report string
        """
        analysis = StorageAnalyzer.analyze_storage_result(result)
        
        report = f"""
Storage Operation Report
========================

Summary:
- Total chunks: {result.total_count}
- Successfully stored: {result.stored_count}
- Failed: {result.error_count}
- Success rate: {result.success_rate:.1%}
- Processing time: {result.processing_time:.2f}s

Performance:
- Chunks per second: {analysis['performance']['chunks_per_second']:.1f}
- Average time per chunk: {analysis['performance']['average_time_per_chunk']:.3f}s

Index Information:
- Index name: {result.index_info.get('index_name', 'N/A')}
- Knowledge base ID: {result.index_info.get('kb_id', 'N/A')}
- Vector size: {result.index_info.get('vector_size', 'N/A')}
- Index exists: {result.index_info.get('index_exists', 'N/A')}

Status: {'✅ SUCCESS' if result.is_successful else '❌ FAILED'}
"""
        
        if result.error_messages:
            report += f"\nErrors (first 5):\n"
            for i, error in enumerate(result.error_messages[:5], 1):
                report += f"  {i}. {error}\n"
        
        if analysis["recommendations"]:
            report += f"\nRecommendations:\n"
            for i, rec in enumerate(analysis["recommendations"], 1):
                report += f"  {i}. {rec}\n"
        
        return report


class StorageExporter:
    """Exporter for storage configurations and results"""
    
    @staticmethod
    def export_index_mapping(index_name: str, output_path: Path):
        """
        Export index mapping information
        
        Args:
            index_name: Name of the index
            output_path: Output file path
        """
        # This would require connecting to ES to get actual mapping
        # For now, we'll export the expected mapping structure
        
        mapping_info = {
            "index_name": index_name,
            "expected_fields": {
                "id": "keyword",
                "doc_id": "keyword", 
                "kb_id": "keyword",
                "content_with_weight": "text",
                "content_ltks": "text (whitespace analyzer)",
                "content_sm_ltks": "text (whitespace analyzer)",
                "docnm_kwd": "keyword",
                "title_tks": "text",
                "title_sm_tks": "text",
                "page_num_int": "integer array",
                "position_int": "integer array",
                "top_int": "integer array",
                "create_time": "date",
                "create_timestamp_flt": "float",
                "q_*_vec": "dense_vector (cosine similarity)"
            },
            "vector_dimensions": [512, 768, 1024, 1536],
            "note": "Actual mapping may vary based on RAGFlow configuration"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_sample_chunk(chunks: List[Dict[str, Any]], output_path: Path):
        """
        Export a sample chunk structure
        
        Args:
            chunks: List of chunks
            output_path: Output file path
        """
        if not chunks:
            sample = {"error": "No chunks provided"}
        else:
            sample = chunks[0].copy()
            
            # Truncate vector for readability
            for key in list(sample.keys()):
                if key.startswith("q_") and key.endswith("_vec"):
                    vector = sample[key]
                    if isinstance(vector, list) and len(vector) > 5:
                        sample[key] = vector[:3] + ["..."] + [f"({len(vector)} dimensions total)"]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
