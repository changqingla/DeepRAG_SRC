#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepRAG 文档分块器

本模块基于 DeepRAG 的深度文档理解算法提供完整的文档分块功能。
它复用了整个 DeepRAG 处理流水线，包括 OCR、版面识别、表格结构识别和智能分块策略。

主要功能：
1. 支持多种文档类型（PDF、Word、Excel、PowerPoint、Markdown等）
2. 智能版面识别和内容提取
3. 基于语义的智能分块
4. 支持表格和图像的结构化处理
5. 多种解析器适配不同文档类型
6. 异步和批量处理支持

作者:   HU TAO
许可证: Apache 2.0
"""

import os
import sys
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import asyncio
import trio
from timeit import default_timer as timer

# 添加 DeepRAG 根目录到 Python 路径
current_dir = Path(__file__).parent.absolute()
DeepRAG_root = current_dir.parent
sys.path.insert(0, str(DeepRAG_root))

# 导入 DeepRAG 核心组件
from rag.app import (
    naive, paper, book, presentation, manual, laws, qa, table,
     one, email, tag
)
from rag.nlp import rag_tokenizer  # RAG 分词器
from rag.utils import num_tokens_from_string  # Token 计数工具
from rag.utils import get_project_base_directory, ParserType  # 工具函数和解析器类型



class DocumentChunker:
    """
    基于 DeepRAG 的文档分块器

    这个类使用 DeepRAG 的先进文档理解算法提供完整的文档分块功能。
    支持多种文档类型和解析策略，能够智能识别文档结构并进行语义分块。

    主要特性：
    - 支持 PDF、Word、Excel、PowerPoint、Markdown 等多种格式
    - 智能版面识别和内容提取
    - 基于语义的分块策略
    - 表格和图像的结构化处理
    - 多种专业领域的解析器（学术论文、法律文档、技术手册等）
    """

    # 解析器工厂映射 - 与 DeepRAG 的 task_executor.py 保持一致
    PARSER_FACTORY = {
        "general": naive,           # 通用解析器
        ParserType.NAIVE: naive,    # 简单文本解析器
        ParserType.PAPER: paper,    # 学术论文解析器
        ParserType.BOOK: book,      # 书籍解析器
        ParserType.PRESENTATION: presentation,  # 演示文稿解析器
        ParserType.MANUAL: manual,  # 技术手册解析器
        ParserType.LAWS: laws,      # 法律文档解析器
        ParserType.QA: qa,          # 问答文档解析器
        ParserType.TABLE: table,    # 表格解析器
        ParserType.ONE: one,        # 单页文档解析器
        ParserType.EMAIL: email,    # 邮件解析器
        ParserType.KG: naive,       # 知识图谱解析器
        ParserType.TAG: tag         # 标签解析器
    }
    
    def __init__(self,
                 parser_type: str = "general",
                 chunk_token_num: int = 256,
                 delimiter: str = "\n。；！？",
                 language: str = "Chinese",
                 layout_recognize: str = "DeepDOC",
                 zoomin: int = 3,
                 from_page: int = 0,
                 to_page: int = 100000):
        """
        初始化文档分块器

        Args:
            parser_type (str): 解析器类型 (general, paper, book 等)
            chunk_token_num (int): 每个分块的最大 token 数量
            delimiter (str): 文本分割符，用于分块边界识别
            language (str): 文档语言 (Chinese/English)
            layout_recognize (str): 版面识别方法 (DeepDOC/Plain Text)
            zoomin (int): OCR 缩放因子，影响图像识别精度
            from_page (int): 起始页码（从0开始）
            to_page (int): 结束页码
        """
        self.parser_type = parser_type.lower()  # 统一转为小写
        self.chunk_token_num = chunk_token_num
        self.delimiter = delimiter
        self.language = language
        self.layout_recognize = layout_recognize
        self.zoomin = zoomin
        self.from_page = from_page
        self.to_page = to_page

        # 验证解析器类型是否支持
        if self.parser_type not in self.PARSER_FACTORY:
            raise ValueError(f"不支持的解析器类型: {parser_type}. "
                           f"支持的类型: {list(self.PARSER_FACTORY.keys())}")

        # 获取对应的解析器实例
        self.chunker = self.PARSER_FACTORY[self.parser_type]

        # 设置日志配置
        self._setup_logging()

        # 解析器配置参数
        self.parser_config = {
            "chunk_token_num": self.chunk_token_num,  # 分块大小
            "delimiter": self.delimiter,              # 分割符
            "layout_recognize": self.layout_recognize # 版面识别方法
        }

        logging.info(f"文档分块器初始化完成，使用解析器: {self.parser_type}")
    
    def _setup_logging(self):
        """
        设置日志配置

        配置日志格式和级别，用于跟踪分块处理过程
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _progress_callback(self, progress: float = None, msg: str = ""):
        """
        分块处理进度回调函数

        用于跟踪文档处理进度，提供实时状态反馈

        Args:
            progress (float): 进度百分比 (0.0-1.0)
            msg (str): 状态消息
        """
        if progress is not None:
            logging.info(f"处理进度: {progress:.1%} - {msg}")
        else:
            logging.info(f"状态: {msg}")
    
    def chunk_document(self,
                      file_path: Union[str, Path],
                      binary_data: Optional[bytes] = None,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        使用 DeepRAG 完整处理流水线对文档进行分块

        这是核心的文档分块方法，集成了 OCR、版面识别、表格识别和智能分块等功能。

        处理流程：
        1. 读取文档二进制数据
        2. 根据文档类型选择合适的解析器
        3. 执行 OCR 和版面识别
        4. 提取文本和结构化内容
        5. 进行智能分块处理
        6. 返回带有元数据的分块结果

        Args:
            file_path (Union[str, Path]): 文档文件路径
            binary_data (Optional[bytes]): 文档二进制数据（可选，如果提供则不读取文件）
            **kwargs: 特定解析器的额外参数

        Returns:
            List[Dict[str, Any]]: 文档分块列表，每个分块包含内容和元数据
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        filename = file_path.name

        logging.info(f"开始对文档进行分块处理: {filename}")
        logging.info(f"使用解析器: {self.parser_type}")

        start_time = timer()

        try:
            # 如果未提供二进制数据，则从文件读取
            if binary_data is None:
                if not file_path.exists():
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                with open(file_path, 'rb') as f:
                    binary_data = f.read()

            # 准备分块处理参数
            chunk_params = {
                'filename': filename,                    # 文件名
                'binary': binary_data,                   # 二进制数据
                'from_page': self.from_page,            # 起始页码
                'to_page': self.to_page,                # 结束页码
                'lang': self.language,                   # 文档语言
                'callback': self._progress_callback,     # 进度回调函数
                'parser_config': self.parser_config,     # 解析器配置
                **kwargs                                 # 额外参数
            }

            # 使用 DeepRAG 算法执行分块处理
            self._progress_callback(0.0, "开始文档处理...")
            chunks = self.chunker.chunk(**chunk_params)

            processing_time = timer() - start_time
            logging.info(f"文档分块处理完成，耗时 {processing_time:.2f}s")
            logging.info(f"生成了 {len(chunks)} 个分块")

            return chunks

        except Exception as e:
            logging.error(f"文档分块处理出错 {filename}: {str(e)}")
            raise
    
    async def chunk_document_async(self,
                                  file_path: Union[str, Path],
                                  binary_data: Optional[bytes] = None,
                                  **kwargs) -> List[Dict[str, Any]]:
        """
        异步版本的文档分块处理

        使用 trio 库在后台线程中执行分块处理，避免阻塞主线程。
        适用于需要处理大量文档或在 Web 应用中使用的场景。

        Args:
            file_path (Union[str, Path]): 文档文件路径
            binary_data (Optional[bytes]): 文档二进制数据（可选）
            **kwargs: 特定解析器的额外参数

        Returns:
            List[Dict[str, Any]]: 文档分块列表，每个分块包含内容和元数据
        """
        return await trio.to_thread.run_sync(
            lambda: self.chunk_document(file_path, binary_data, **kwargs)
        )

    def chunk_batch(self,
                   file_paths: List[Union[str, Path]],
                   **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量处理多个文档的分块

        按顺序处理多个文档文件，对每个文件执行分块操作。
        即使某个文件处理失败，也会继续处理其他文件。

        Args:
            file_paths (List[Union[str, Path]]): 要处理的文件路径列表
            **kwargs: 分块处理的额外参数

        Returns:
            Dict[str, List[Dict[str, Any]]]: 文件路径到分块结果的映射字典
        """
        results = {}
        total_files = len(file_paths)

        logging.info(f"开始批量分块处理，共 {total_files} 个文件")

        for i, file_path in enumerate(file_paths):
            try:
                logging.info(f"正在处理文件 {i+1}/{total_files}: {file_path}")
                chunks = self.chunk_document(file_path, **kwargs)
                results[str(file_path)] = chunks
            except Exception as e:
                logging.error(f"处理文件失败 {file_path}: {str(e)}")
                results[str(file_path)] = []  # 失败时返回空列表

        logging.info(f"批量分块处理完成，已处理 {len(results)} 个文件")
        return results

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取分块统计信息

        分析分块结果，提供详细的统计数据，包括分块数量、token 分布、
        字符数等信息，用于评估分块质量和优化参数。

        Args:
            chunks (List[Dict[str, Any]]): 文档分块列表

        Returns:
            Dict[str, Any]: 包含分块统计信息的字典
                - total_chunks: 总分块数
                - total_tokens: 总 token 数
                - avg_tokens_per_chunk: 平均每个分块的 token 数
                - min_tokens: 最小分块的 token 数
                - max_tokens: 最大分块的 token 数
                - total_characters: 总字符数
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "total_characters": 0
            }

        token_counts = []
        char_counts = []

        # 遍历所有分块，统计 token 和字符数
        for chunk in chunks:
            content = chunk.get("content_with_weight", "")
            if content:
                tokens = num_tokens_from_string(content)
                token_counts.append(tokens)
                char_counts.append(len(content))

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "total_characters": sum(char_counts)
        }

    def export_chunks(self,
                     chunks: List[Dict[str, Any]],
                     output_path: Union[str, Path],
                     format: str = "json") -> None:
        """
        将分块结果导出到文件

        支持多种导出格式，方便后续处理和分析。

        Args:
            chunks (List[Dict[str, Any]]): 文档分块列表
            output_path (Union[str, Path]): 输出文件路径
            format (str): 导出格式 (json, txt, csv)
                - json: 保留完整的元数据信息
                - txt: 纯文本格式，便于阅读
                - csv: 表格格式，便于数据分析
        """
        import json
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录

        if format.lower() == "json":
            # JSON 格式：保留所有元数据
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        elif format.lower() == "txt":
            # 文本格式：仅保存内容，便于阅读
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    content = chunk.get("content_with_weight", "")
                    f.write(f"=== 分块 {i+1} ===\n")
                    f.write(content)
                    f.write("\n\n")

        elif format.lower() == "csv":
            # CSV 格式：结构化数据，便于分析
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["分块ID", "内容", "Token数", "文档名"])

                for i, chunk in enumerate(chunks):
                    content = chunk.get("content_with_weight", "")
                    tokens = num_tokens_from_string(content)
                    doc_name = chunk.get("docnm_kwd", "")
                    writer.writerow([i+1, content, tokens, doc_name])

        else:
            raise ValueError(f"不支持的导出格式: {format}")

        logging.info(f"分块结果已导出到 {output_path}，格式: {format}")

    @classmethod
    def get_supported_parsers(cls) -> List[str]:
        """
        获取支持的解析器类型列表

        Returns:
            List[str]: 支持的解析器类型列表
        """
        return list(cls.PARSER_FACTORY.keys())

    @classmethod
    def get_parser_info(cls, parser_type: str) -> Dict[str, Any]:
        """
        获取特定解析器的详细信息

        提供解析器的描述、适用场景和模块信息，帮助用户选择合适的解析器。

        Args:
            parser_type (str): 解析器类型

        Returns:
            Dict[str, Any]: 包含解析器信息的字典
                - name: 解析器名称
                - description: 解析器描述
                - module: 解析器模块名
        """
        parser_descriptions = {
            "general": "通用解析器，适用于常见文档类型",
            "naive": "简单文本解析器，基础的文本提取功能",
            "paper": "学术论文解析器，具有高级版面识别功能",
            "book": "书籍解析器，支持章节和段落检测",
            "presentation": "演示文稿解析器，适用于 PowerPoint 等格式",
            "manual": "技术手册解析器，适用于技术文档",
            "laws": "法律文档解析器，专门处理法律条文",
            "qa": "问答文档解析器，适用于问答格式的文档",
            "table": "表格专用解析器，专注于表格内容提取",
            "one": "单页文档解析器，适用于简单的单页文档",
            "email": "邮件文档解析器，处理邮件格式",
            "kg": "知识图谱解析器，用于构建知识图谱",
            "tag": "标签解析器，基于标签的文档处理"
        }

        if parser_type not in cls.PARSER_FACTORY:
            raise ValueError(f"未知的解析器类型: {parser_type}")

        return {
            "name": parser_type,
            "description": parser_descriptions.get(parser_type, "暂无描述"),
            "module": cls.PARSER_FACTORY[parser_type].__name__
        }

    @staticmethod
    def detect_document_type(file_path: Union[str, Path]) -> str:
        """
        根据文件扩展名检测文档类型

        Args:
            file_path (Union[str, Path]): 文件路径

        Returns:
            str: 文档类型 (pdf, docx, txt, etc.)
        """
        file_path = Path(file_path)
        return file_path.suffix.lower().lstrip('.')

    @staticmethod
    def recommend_parser(file_path: Union[str, Path]) -> str:
        """
        根据文件类型推荐合适的解析器

        Args:
            file_path (Union[str, Path]): 文件路径

        Returns:
            str: 推荐的解析器类型
        """
        doc_type = DocumentChunker.detect_document_type(file_path)
        filename = Path(file_path).name.lower()

        # 根据文件名特征推荐解析器
        if any(keyword in filename for keyword in ['paper', 'journal', 'article', 'research']):
            return "paper"
        elif any(keyword in filename for keyword in ['manual', 'guide', 'handbook']):
            return "manual"
        elif any(keyword in filename for keyword in ['law', 'legal', 'regulation']):
            return "laws"
        elif any(keyword in filename for keyword in ['qa', 'faq', 'question']):
            return "qa"
        elif doc_type in ['ppt', 'pptx']:
            return "presentation"
        elif doc_type in ['xls', 'xlsx']:
            return "table"
        else:
            return "general"

    def validate_parameters(self) -> Dict[str, Any]:
        """
        验证分块器参数的有效性

        Returns:
            Dict[str, Any]: 验证结果，包含是否有效和错误信息
        """
        errors = []
        warnings = []

        # 检查 chunk_token_num
        if self.chunk_token_num < 50:
            warnings.append("chunk_token_num 过小可能导致分块过于细碎")
        elif self.chunk_token_num > 2048:
            warnings.append("chunk_token_num 过大可能影响检索精度")

        # 检查页码范围
        if self.from_page < 0:
            errors.append("from_page 不能为负数")
        if self.to_page <= self.from_page:
            errors.append("to_page 必须大于 from_page")

        # 检查语言设置
        if self.language not in ["Chinese", "English"]:
            warnings.append(f"语言设置 '{self.language}' 可能不被完全支持")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# 使用示例
if __name__ == "__main__":
    """
    DocumentChunker 使用示例

    演示如何使用文档分块器处理不同类型的文档
    """
    import logging

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 示例1: 基本使用
    print("=== 基本使用示例 ===")
    chunker = DocumentChunker(
        parser_type="general",
        chunk_token_num=256,
        language="Chinese"
    )

    # 验证参数
    validation = chunker.validate_parameters()
    if not validation["valid"]:
        print(f"参数验证失败: {validation['errors']}")
    if validation["warnings"]:
        print(f"警告: {validation['warnings']}")

    # 示例2: 获取支持的解析器
    print("\n=== 支持的解析器 ===")
    parsers = DocumentChunker.get_supported_parsers()
    for parser in parsers[:5]:  # 只显示前5个
        info = DocumentChunker.get_parser_info(parser)
        print(f"- {info['name']}: {info['description']}")

    # 示例3: 文档类型检测和解析器推荐
    print("\n=== 文档类型检测示例 ===")
    test_files = [
        "research_paper.pdf",
        "user_manual.docx",
        "legal_document.pdf",
        "presentation.pptx"
    ]

    for file_path in test_files:
        doc_type = DocumentChunker.detect_document_type(file_path)
        recommended = DocumentChunker.recommend_parser(file_path)
        print(f"{file_path}: 类型={doc_type}, 推荐解析器={recommended}")

    print("\n=== 使用说明 ===")
    print("1. 根据文档类型选择合适的解析器")
    print("2. 调整 chunk_token_num 参数控制分块大小")
    print("3. 使用 chunk_document() 方法处理单个文档")
    print("4. 使用 chunk_batch() 方法批量处理文档")
    print("5. 使用 export_chunks() 方法导出结果")
