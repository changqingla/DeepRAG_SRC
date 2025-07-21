#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepRAG 文档分块器工具模块

本模块为基于 DeepRAG 的文档分块器提供实用工具函数和辅助类，包括：
- 文件类型检测和解析器推荐
- 分块配置管理
- 分块结果分析和质量评估
- 结果格式化和报告生成

主要功能：
1. ChunkingConfig: 分块配置管理类
2. FileTypeDetector: 文件类型检测和解析器推荐
3. ChunkAnalyzer: 分块结果质量分析
4. ResultFormatter: 结果格式化和报告生成

作者:   HU TAO
许可证: Apache 2.0
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
from dataclasses import dataclass, asdict


@dataclass
class ChunkingConfig:
    """
    文档分块配置类

    使用数据类管理文档分块的所有配置参数，提供配置的序列化、
    反序列化和文件存储功能。

    属性:
        parser_type (str): 解析器类型，默认为 "general"
        chunk_token_num (int): 每个分块的最大 token 数量，默认 256
        delimiter (str): 文本分割符，用于识别分块边界
        language (str): 文档语言，影响分词和处理策略
        layout_recognize (str): 版面识别方法，默认 "DeepDOC"
        zoomin (int): OCR 缩放因子，影响图像识别精度
        from_page (int): 起始页码，从 0 开始
        to_page (int): 结束页码，默认处理所有页面
    """
    parser_type: str = "general"           # 解析器类型
    chunk_token_num: int = 256             # 分块大小（token数）
    delimiter: str = "\n。；！？"          # 文本分割符
    language: str = "Chinese"              # 文档语言
    layout_recognize: str = "DeepDOC"      # 版面识别方法
    zoomin: int = 3                        # OCR 缩放因子
    from_page: int = 0                     # 起始页码
    to_page: int = 100000                  # 结束页码

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式

        Returns:
            Dict[str, Any]: 配置字典
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChunkingConfig':
        """
        从字典创建配置对象

        Args:
            config_dict (Dict[str, Any]): 配置字典

        Returns:
            ChunkingConfig: 配置对象实例
        """
        return cls(**config_dict)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        将配置保存到 JSON 文件

        Args:
            file_path (Union[str, Path]): 文件保存路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ChunkingConfig':
        """
        从 JSON 文件加载配置

        Args:
            file_path (Union[str, Path]): 配置文件路径

        Returns:
            ChunkingConfig: 加载的配置对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证配置参数的有效性

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []

        # 验证 chunk_token_num
        if self.chunk_token_num < 1:
            errors.append("chunk_token_num 必须大于 0")
        elif self.chunk_token_num > 4096:
            errors.append("chunk_token_num 不应超过 4096")

        # 验证页码范围
        if self.from_page < 0:
            errors.append("from_page 不能为负数")
        if self.to_page <= self.from_page:
            errors.append("to_page 必须大于 from_page")

        # 验证缩放因子
        if self.zoomin < 1 or self.zoomin > 10:
            errors.append("zoomin 应在 1-10 范围内")

        return len(errors) == 0, errors


class FileTypeDetector:
    """
    文件类型检测和解析器推荐工具类

    提供文件类型自动检测和解析器推荐功能，支持多种文档格式。
    通过文件扩展名和 MIME 类型双重检测，提高识别准确性。
    """

    # 文件扩展名到解析器类型的映射表
    # 按推荐优先级排序，第一个为最佳推荐
    EXTENSION_PARSER_MAP = {
        '.pdf': ['paper', 'book', 'manual', 'general'],    # PDF文档
        '.docx': ['book', 'manual', 'general'],            # Word文档
        '.doc': ['book', 'manual', 'general'],             # 旧版Word文档
        '.txt': ['general', 'naive'],                      # 纯文本文件
        '.md': ['general', 'naive'],                       # Markdown文件
        '.html': ['general'],                              # HTML文件
        '.htm': ['general'],                               # HTML文件
        '.xlsx': ['table'],                                # Excel表格
        '.xls': ['table'],                                 # 旧版Excel表格
        '.csv': ['table'],                                 # CSV表格
        '.pptx': ['presentation'],                         # PowerPoint演示文稿
        '.ppt': ['presentation'],                          # 旧版PowerPoint
        '.jpg': ['picture'],                               # JPEG图片
        '.jpeg': ['picture'],                              # JPEG图片
        '.png': ['picture'],                               # PNG图片
        '.bmp': ['picture'],                               # BMP图片
        '.tiff': ['picture'],                              # TIFF图片
        '.mp3': ['audio'],                                 # MP3音频
        '.wav': ['audio'],                                 # WAV音频
        '.m4a': ['audio'],                                 # M4A音频
        '.json': ['general'],                              # JSON数据文件
        '.xml': ['general']                                # XML数据文件
    }
    
    @classmethod
    def detect_file_type(cls, file_path: Union[str, Path]) -> str:
        """
        基于文件扩展名和 MIME 类型检测文件类型

        使用双重检测机制提高文件类型识别的准确性：
        1. 首先检查文件扩展名
        2. 如果扩展名不在支持列表中，则使用 MIME 类型检测

        Args:
            file_path (Union[str, Path]): 文件路径

        Returns:
            str: 检测到的文件类型（扩展名格式，如 '.pdf'）
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        # 首先尝试通过扩展名识别
        if extension in cls.EXTENSION_PARSER_MAP:
            return extension

        # 如果扩展名不在支持列表中，尝试 MIME 类型检测
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type:
            # 根据 MIME 类型映射到对应的文件类型
            if mime_type.startswith('text/'):
                return '.txt'
            elif mime_type.startswith('image/'):
                return '.jpg'
            elif mime_type.startswith('audio/'):
                return '.mp3'
            elif 'pdf' in mime_type:
                return '.pdf'
            elif 'word' in mime_type or 'document' in mime_type:
                return '.docx'
            elif 'spreadsheet' in mime_type or 'excel' in mime_type:
                return '.xlsx'
            elif 'presentation' in mime_type or 'powerpoint' in mime_type:
                return '.pptx'

        # 默认回退到文本文件
        return '.txt'
    
    @classmethod
    def recommend_parser(cls, file_path: Union[str, Path]) -> List[str]:
        """
        为指定文件推荐合适的解析器类型

        根据文件类型和文件名特征，返回按优先级排序的解析器推荐列表。
        第一个解析器为最佳推荐。

        Args:
            file_path (Union[str, Path]): 文件路径

        Returns:
            List[str]: 推荐的解析器类型列表（按优先级排序）
        """
        file_type = cls.detect_file_type(file_path)
        filename = Path(file_path).name.lower()

        # 获取基础推荐列表
        base_recommendations = cls.EXTENSION_PARSER_MAP.get(file_type, ['general'])

        # 根据文件名特征进行优化推荐
        if any(keyword in filename for keyword in ['paper', 'journal', 'article', 'research']):
            # 学术论文相关
            if 'paper' in base_recommendations:
                base_recommendations = ['paper'] + [p for p in base_recommendations if p != 'paper']
        elif any(keyword in filename for keyword in ['manual', 'guide', 'handbook', 'doc']):
            # 技术手册相关
            if 'manual' in base_recommendations:
                base_recommendations = ['manual'] + [p for p in base_recommendations if p != 'manual']
        elif any(keyword in filename for keyword in ['law', 'legal', 'regulation', 'contract']):
            # 法律文档相关
            base_recommendations = ['laws'] + base_recommendations
        elif any(keyword in filename for keyword in ['qa', 'faq', 'question', 'answer']):
            # 问答文档相关
            base_recommendations = ['qa'] + base_recommendations

        return base_recommendations

    @classmethod
    def is_supported_file(cls, file_path: Union[str, Path]) -> bool:
        """
        检查文件类型是否受支持

        Args:
            file_path (Union[str, Path]): 文件路径

        Returns:
            bool: 如果文件类型受支持则返回 True
        """
        file_type = cls.detect_file_type(file_path)
        return file_type in cls.EXTENSION_PARSER_MAP

    @classmethod
    def get_file_info(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件的详细信息

        Args:
            file_path (Union[str, Path]): 文件路径

        Returns:
            Dict[str, Any]: 文件信息字典
        """
        file_path = Path(file_path)
        file_type = cls.detect_file_type(file_path)

        info = {
            "filename": file_path.name,
            "file_type": file_type,
            "is_supported": file_type in cls.EXTENSION_PARSER_MAP,
            "recommended_parsers": cls.recommend_parser(file_path),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "mime_type": mimetypes.guess_type(str(file_path))[0]
        }

        return info


class ChunkAnalyzer:
    """
    分块结果分析工具类

    提供分块质量分析、统计计算和优化建议功能。
    通过多维度指标评估分块效果，帮助优化分块参数。
    """

    @staticmethod
    def analyze_chunk_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析分块质量

        通过多个维度评估分块质量，包括：
        - 分块数量和大小分布
        - 内容长度和 token 数量统计
        - 异常分块检测（过短或过长）
        - 综合质量评分

        Args:
            chunks (List[Dict[str, Any]]): 文档分块列表

        Returns:
            Dict[str, Any]: 包含质量指标的字典
        """
        if not chunks:
            return {"error": "没有分块数据可供分析"}

        # 提取内容长度和 token 数量
        content_lengths = []
        token_counts = []
        empty_chunks = 0

        for chunk in chunks:
            content = chunk.get("content_with_weight", "")
            if content:
                content_lengths.append(len(content))
                # 估算 token 数量（粗略近似）
                token_count = len(content.split())
                token_counts.append(token_count)
            else:
                empty_chunks += 1

        if not content_lengths:
            return {"error": "分块中未找到有效内容"}

        # 计算基础统计信息
        avg_length = sum(content_lengths) / len(content_lengths)
        avg_tokens = sum(token_counts) / len(token_counts)
        min_length = min(content_lengths)
        max_length = max(content_lengths)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)

        # 计算方差（衡量分块大小的一致性）
        length_variance = sum((x - avg_length) ** 2 for x in content_lengths) / len(content_lengths)
        token_variance = sum((x - avg_tokens) ** 2 for x in token_counts) / len(token_counts)

        # 检测异常分块
        short_chunks = sum(1 for length in content_lengths if length < 50)      # 过短分块
        long_chunks = sum(1 for length in content_lengths if length > 2000)    # 过长分块

        return {
            "total_chunks": len(chunks),                                        # 总分块数
            "empty_chunks": empty_chunks,                                       # 空分块数
            "avg_content_length": avg_length,                                   # 平均内容长度
            "avg_token_count": avg_tokens,                                      # 平均 token 数
            "min_content_length": min_length,                                   # 最小内容长度
            "max_content_length": max_length,                                   # 最大内容长度
            "min_token_count": min_tokens,                                      # 最小 token 数
            "max_token_count": max_tokens,                                      # 最大 token 数
            "length_variance": length_variance,                                 # 长度方差
            "token_variance": token_variance,                                   # token 方差
            "short_chunks_count": short_chunks,                                 # 过短分块数量
            "long_chunks_count": long_chunks,                                   # 过长分块数量
            "short_chunks_ratio": short_chunks / len(chunks),                   # 过短分块比例
            "long_chunks_ratio": long_chunks / len(chunks),                     # 过长分块比例
            "quality_score": ChunkAnalyzer._calculate_quality_score(            # 综合质量评分
                short_chunks / len(chunks),
                long_chunks / len(chunks),
                length_variance / (avg_length ** 2) if avg_length > 0 else 0
            )
        }
    
    @staticmethod
    def _calculate_quality_score(short_ratio: float, long_ratio: float,
                               normalized_variance: float) -> float:
        """
        计算分块质量评分 (0-100)

        基于异常分块比例和长度方差计算综合质量评分。
        评分越高表示分块质量越好。

        评分规则：
        - 过短分块比例高：扣分较多（影响内容完整性）
        - 过长分块比例高：扣分中等（影响检索精度）
        - 长度方差大：扣分较多（分块不均匀）

        Args:
            short_ratio (float): 过短分块的比例
            long_ratio (float): 过长分块的比例
            normalized_variance (float): 标准化的长度方差

        Returns:
            float: 质量评分（0-100，越高越好）
        """
        # 计算惩罚分数
        penalty = (short_ratio * 30) + (long_ratio * 20) + (normalized_variance * 50)
        score = max(0, 100 - penalty)
        return score

    @staticmethod
    def find_optimal_chunk_size(sample_text: str,
                              target_chunks: int = 10) -> int:
        """
        为给定文本找到最优的分块大小

        基于文本长度和目标分块数量，计算推荐的分块大小。
        考虑实际使用场景，设置合理的上下界限。

        Args:
            sample_text (str): 样本文本
            target_chunks (int): 目标分块数量，默认 10

        Returns:
            int: 推荐的分块 token 大小
        """
        # 估算总 token 数
        total_tokens = len(sample_text.split())

        # 计算最优分块大小
        optimal_size = total_tokens // target_chunks

        # 应用合理的边界限制
        # 最小 128 tokens：保证分块有足够的上下文
        # 最大 512 tokens：避免分块过大影响检索精度
        optimal_size = max(128, min(optimal_size, 512))

        return optimal_size

    @staticmethod
    def suggest_improvements(quality_metrics: Dict[str, Any]) -> List[str]:
        """
        基于质量指标提供改进建议

        Args:
            quality_metrics (Dict[str, Any]): 质量分析结果

        Returns:
            List[str]: 改进建议列表
        """
        suggestions = []

        # 检查质量评分
        quality_score = quality_metrics.get('quality_score', 0)
        if quality_score < 60:
            suggestions.append("整体分块质量较低，建议调整分块参数")

        # 检查过短分块
        short_ratio = quality_metrics.get('short_chunks_ratio', 0)
        if short_ratio > 0.2:
            suggestions.append("过短分块比例过高，建议减小 chunk_token_num 或调整分割符")

        # 检查过长分块
        long_ratio = quality_metrics.get('long_chunks_ratio', 0)
        if long_ratio > 0.1:
            suggestions.append("过长分块比例过高，建议增大 chunk_token_num")

        # 检查方差
        length_variance = quality_metrics.get('length_variance', 0)
        avg_length = quality_metrics.get('avg_content_length', 1)
        if length_variance / (avg_length ** 2) > 0.5:
            suggestions.append("分块大小不均匀，建议优化分割策略")

        # 检查空分块
        empty_chunks = quality_metrics.get('empty_chunks', 0)
        if empty_chunks > 0:
            suggestions.append(f"发现 {empty_chunks} 个空分块，建议检查文档内容")

        if not suggestions:
            suggestions.append("分块质量良好，无需调整")

        return suggestions


class ResultFormatter:
    """
    分块结果格式化工具类

    提供多种格式化功能，将分块结果转换为适合显示、
    分析或导出的格式。
    """

    @staticmethod
    def format_chunks_for_display(chunks: List[Dict[str, Any]],
                                max_content_length: int = 200) -> List[Dict[str, Any]]:
        """
        格式化分块结果用于显示

        将原始分块数据转换为适合前端显示的格式，包括：
        - 内容预览（截断过长内容）
        - 基础统计信息
        - 元数据提取

        Args:
            chunks (List[Dict[str, Any]]): 原始分块列表
            max_content_length (int): 显示内容的最大长度，默认 200

        Returns:
            List[Dict[str, Any]]: 格式化后的分块列表
        """
        formatted = []

        for i, chunk in enumerate(chunks):
            content = chunk.get("content_with_weight", "")

            # 截断过长的内容用于预览
            if len(content) > max_content_length:
                display_content = content[:max_content_length] + "..."
            else:
                display_content = content

            # 构建格式化的分块信息
            formatted_chunk = {
                "chunk_id": i + 1,                                      # 分块ID
                "content_preview": display_content,                     # 内容预览
                "content_length": len(content),                         # 内容长度
                "token_count": len(content.split()),                    # 估算token数
                "doc_name": chunk.get("docnm_kwd", "未知文档"),         # 文档名
                "has_image": "image" in chunk,                          # 是否包含图像
                "has_table": any("table" in str(v).lower() for v in chunk.values()),  # 是否包含表格
                "metadata": {k: v for k, v in chunk.items()             # 其他元数据
                           if k not in ["content_with_weight", "image"]}
            }

            formatted.append(formatted_chunk)

        return formatted
    
    @staticmethod
    def create_summary_report(chunks: List[Dict[str, Any]],
                            processing_time: float = None,
                            config: ChunkingConfig = None) -> str:
        """
        创建分块结果摘要报告

        生成详细的分块处理报告，包括统计信息、质量分析和改进建议。

        Args:
            chunks (List[Dict[str, Any]]): 分块列表
            processing_time (float): 处理耗时（秒）
            config (ChunkingConfig): 分块配置信息

        Returns:
            str: 格式化的摘要报告
        """
        if not chunks:
            return "未生成任何分块。"

        # 基础统计信息
        total_chunks = len(chunks)
        total_content = sum(len(chunk.get("content_with_weight", "")) for chunk in chunks)
        avg_chunk_size = total_content / total_chunks if total_chunks > 0 else 0

        # 质量分析
        quality_metrics = ChunkAnalyzer.analyze_chunk_quality(chunks)
        suggestions = ChunkAnalyzer.suggest_improvements(quality_metrics)

        # 构建报告
        report = f"""
文档分块处理摘要报告
==================

基础统计信息:
- 生成分块总数: {total_chunks}
- 内容总长度: {total_content:,} 字符
- 平均分块大小: {avg_chunk_size:.1f} 字符
- 质量评分: {quality_metrics.get('quality_score', 0):.1f}/100

质量指标:
- 过短分块 (< 50字符): {quality_metrics.get('short_chunks_count', 0)} 个 ({quality_metrics.get('short_chunks_ratio', 0):.1%})
- 过长分块 (> 2000字符): {quality_metrics.get('long_chunks_count', 0)} 个 ({quality_metrics.get('long_chunks_ratio', 0):.1%})
- 空分块: {quality_metrics.get('empty_chunks', 0)} 个
- 内容长度方差: {quality_metrics.get('length_variance', 0):.1f}
- 平均token数: {quality_metrics.get('avg_token_count', 0):.1f}

分块大小分布:
- 最小长度: {quality_metrics.get('min_content_length', 0)} 字符
- 最大长度: {quality_metrics.get('max_content_length', 0)} 字符
- 最小token数: {quality_metrics.get('min_token_count', 0)}
- 最大token数: {quality_metrics.get('max_token_count', 0)}
"""

        # 添加配置信息
        if config:
            report += f"""
使用的配置参数:
- 解析器类型: {config.parser_type}
- 分块大小: {config.chunk_token_num} tokens
- 分割符: {config.delimiter}
- 文档语言: {config.language}
- 版面识别: {config.layout_recognize}
"""

        # 添加处理时间
        if processing_time:
            report += f"\n处理耗时: {processing_time:.2f} 秒"

        # 添加改进建议
        report += f"""

改进建议:
"""
        for i, suggestion in enumerate(suggestions, 1):
            report += f"{i}. {suggestion}\n"

        return report

    @staticmethod
    def export_to_json(chunks: List[Dict[str, Any]],
                      output_path: Union[str, Path],
                      include_metadata: bool = True) -> None:
        """
        将分块结果导出为 JSON 格式

        Args:
            chunks (List[Dict[str, Any]]): 分块列表
            output_path (Union[str, Path]): 输出文件路径
            include_metadata (bool): 是否包含元数据
        """
        import json

        if not include_metadata:
            # 只保留核心内容
            simplified_chunks = [
                {
                    "id": i + 1,
                    "content": chunk.get("content_with_weight", ""),
                    "doc_name": chunk.get("docnm_kwd", "")
                }
                for i, chunk in enumerate(chunks)
            ]
            data = simplified_chunks
        else:
            data = chunks

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_to_csv(chunks: List[Dict[str, Any]],
                     output_path: Union[str, Path]) -> None:
        """
        将分块结果导出为 CSV 格式

        Args:
            chunks (List[Dict[str, Any]]): 分块列表
            output_path (Union[str, Path]): 输出文件路径
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["分块ID", "内容", "字符数", "Token数", "文档名"])

            for i, chunk in enumerate(chunks):
                content = chunk.get("content_with_weight", "")
                writer.writerow([
                    i + 1,
                    content,
                    len(content),
                    len(content.split()),
                    chunk.get("docnm_kwd", "")
                ])


# 使用示例
if __name__ == "__main__":
    """
    ChunkerUtils 使用示例

    演示各个工具类的基本用法
    """
    import logging

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    print("=== DeepRAG 文档分块器工具示例 ===\n")

    # 示例1: 配置管理
    print("1. 配置管理示例:")
    config = ChunkingConfig(
        parser_type="paper",
        chunk_token_num=512,
        language="Chinese"
    )

    # 验证配置
    is_valid, errors = config.validate()
    print(f"配置有效性: {is_valid}")
    if errors:
        print(f"错误信息: {errors}")

    # 保存和加载配置
    config_file = "chunk_config.json"
    config.save_to_file(config_file)
    print(f"配置已保存到: {config_file}")

    # 示例2: 文件类型检测
    print("\n2. 文件类型检测示例:")
    test_files = [
        "research_paper.pdf",
        "user_manual.docx",
        "data_table.xlsx",
        "presentation.pptx",
        "unknown_file.xyz"
    ]

    for file_path in test_files:
        file_info = FileTypeDetector.get_file_info(file_path)
        print(f"{file_path}:")
        print(f"  - 文件类型: {file_info['file_type']}")
        print(f"  - 是否支持: {file_info['is_supported']}")
        print(f"  - 推荐解析器: {file_info['recommended_parsers']}")

    # 示例3: 模拟分块质量分析
    print("\n3. 分块质量分析示例:")
    # 模拟分块数据
    mock_chunks = [
        {"content_with_weight": "这是第一个分块的内容，包含了一些重要信息。" * 5},
        {"content_with_weight": "第二个分块"},  # 过短分块
        {"content_with_weight": "这是第三个分块的内容。" * 20},  # 正常分块
        {"content_with_weight": "这是一个很长的分块内容。" * 100},  # 过长分块
        {"content_with_weight": "最后一个分块的内容。" * 3}
    ]

    # 分析质量
    quality_metrics = ChunkAnalyzer.analyze_chunk_quality(mock_chunks)
    print(f"质量评分: {quality_metrics['quality_score']:.1f}/100")
    print(f"总分块数: {quality_metrics['total_chunks']}")
    print(f"平均长度: {quality_metrics['avg_content_length']:.1f} 字符")

    # 获取改进建议
    suggestions = ChunkAnalyzer.suggest_improvements(quality_metrics)
    print("改进建议:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")

    # 示例4: 结果格式化
    print("\n4. 结果格式化示例:")
    formatted_chunks = ResultFormatter.format_chunks_for_display(mock_chunks, max_content_length=50)
    print(f"格式化后的分块数: {len(formatted_chunks)}")
    for chunk in formatted_chunks[:2]:  # 只显示前两个
        print(f"  分块 {chunk['chunk_id']}: {chunk['content_preview']}")

    # 生成摘要报告
    print("\n5. 摘要报告示例:")
    report = ResultFormatter.create_summary_report(
        mock_chunks,
        processing_time=2.5,
        config=config
    )
    print(report)

    print("\n=== 示例完成 ===")
    print("更多功能请参考各个类的文档字符串。")
