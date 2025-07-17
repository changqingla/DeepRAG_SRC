#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow Document Chunker

This module provides a complete document chunking functionality based on RAGFlow's 
deep document understanding algorithms. It reuses the entire RAGFlow processing 
pipeline including OCR, layout recognition, table structure recognition, and 
intelligent chunking strategies.

Author: RAGFlow Dev Team
License: Apache 2.0
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

# Add RAGFlow root to path
current_dir = Path(__file__).parent.absolute()
ragflow_root = current_dir.parent
sys.path.insert(0, str(ragflow_root))

# Import RAGFlow components
from rag.app import (
    naive, paper, book, presentation, manual, laws, qa, table,
     one, email, tag
)
from rag.nlp import rag_tokenizer
from rag.utils import num_tokens_from_string
from rag.utils import get_project_base_directory,ParserType



class DocumentChunker:
    """
    RAGFlow-based Document Chunker
    
    This class provides complete document chunking functionality using RAGFlow's
    sophisticated document understanding algorithms. It supports multiple document
    types and parsing strategies.
    """
    
    # Parser factory mapping - same as RAGFlow's task_executor.py
    PARSER_FACTORY = {
        "general": naive,
        ParserType.NAIVE: naive,
        ParserType.PAPER: paper,
        ParserType.BOOK: book,
        ParserType.PRESENTATION: presentation,
        ParserType.MANUAL: manual,
        ParserType.LAWS: laws,
        ParserType.QA: qa,
        ParserType.TABLE: table,
        ParserType.ONE: one,
        ParserType.EMAIL: email,
        ParserType.KG: naive,
        ParserType.TAG: tag
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
        Initialize the DocumentChunker
        
        Args:
            parser_type: Type of parser to use (general, paper, book, etc.)
            chunk_token_num: Maximum tokens per chunk
            delimiter: Text delimiters for chunking
            language: Document language (Chinese/English)
            layout_recognize: Layout recognition method
            zoomin: Zoom factor for OCR
            from_page: Starting page number
            to_page: Ending page number
        """
        self.parser_type = parser_type.lower()
        self.chunk_token_num = chunk_token_num
        self.delimiter = delimiter
        self.language = language
        self.layout_recognize = layout_recognize
        self.zoomin = zoomin
        self.from_page = from_page
        self.to_page = to_page
        
        # Validate parser type
        if self.parser_type not in self.PARSER_FACTORY:
            raise ValueError(f"Unsupported parser type: {parser_type}. "
                           f"Supported types: {list(self.PARSER_FACTORY.keys())}")
        
        self.chunker = self.PARSER_FACTORY[self.parser_type]
        
        # Setup logging
        self._setup_logging()
        
        # Parser configuration
        self.parser_config = {
            "chunk_token_num": self.chunk_token_num,
            "delimiter": self.delimiter,
            "layout_recognize": self.layout_recognize
        }
        
        logging.info(f"DocumentChunker initialized with parser: {self.parser_type}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _progress_callback(self, progress: float = None, msg: str = ""):
        """Progress callback function for chunking process"""
        if progress is not None:
            logging.info(f"Progress: {progress:.1%} - {msg}")
        else:
            logging.info(f"Status: {msg}")
    
    def chunk_document(self, 
                      file_path: Union[str, Path], 
                      binary_data: Optional[bytes] = None,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk a document using RAGFlow's complete processing pipeline
        
        Args:
            file_path: Path to the document file
            binary_data: Binary data of the document (optional)
            **kwargs: Additional parameters for specific parsers
            
        Returns:
            List of document chunks with metadata
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        filename = file_path.name
        
        logging.info(f"Starting document chunking for: {filename}")
        logging.info(f"Using parser: {self.parser_type}")
        
        start_time = timer()
        
        try:
            # Read binary data if not provided
            if binary_data is None:
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
            
            # Prepare chunking parameters
            chunk_params = {
                'filename': filename,
                'binary': binary_data,
                'from_page': self.from_page,
                'to_page': self.to_page,
                'lang': self.language,
                'callback': self._progress_callback,
                'parser_config': self.parser_config,
                **kwargs
            }
            
            # Execute chunking using RAGFlow's algorithm
            self._progress_callback(0.0, "Starting document processing...")
            chunks = self.chunker.chunk(**chunk_params)
            
            processing_time = timer() - start_time
            logging.info(f"Document chunking completed in {processing_time:.2f}s")
            logging.info(f"Generated {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error chunking document {filename}: {str(e)}")
            raise
    
    async def chunk_document_async(self, 
                                  file_path: Union[str, Path], 
                                  binary_data: Optional[bytes] = None,
                                  **kwargs) -> List[Dict[str, Any]]:
        """
        Asynchronous version of document chunking
        
        Args:
            file_path: Path to the document file
            binary_data: Binary data of the document (optional)
            **kwargs: Additional parameters for specific parsers
            
        Returns:
            List of document chunks with metadata
        """
        return await trio.to_thread.run_sync(
            lambda: self.chunk_document(file_path, binary_data, **kwargs)
        )

    def chunk_batch(self,
                   file_paths: List[Union[str, Path]],
                   **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk multiple documents in batch

        Args:
            file_paths: List of file paths to process
            **kwargs: Additional parameters for chunking

        Returns:
            Dictionary mapping file paths to their chunks
        """
        results = {}
        total_files = len(file_paths)

        logging.info(f"Starting batch chunking for {total_files} files")

        for i, file_path in enumerate(file_paths):
            try:
                logging.info(f"Processing file {i+1}/{total_files}: {file_path}")
                chunks = self.chunk_document(file_path, **kwargs)
                results[str(file_path)] = chunks
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {str(e)}")
                results[str(file_path)] = []

        logging.info(f"Batch chunking completed. Processed {len(results)} files")
        return results

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks

        Args:
            chunks: List of document chunks

        Returns:
            Dictionary containing chunk statistics
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
        Export chunks to file

        Args:
            chunks: List of document chunks
            output_path: Output file path
            format: Export format (json, txt, csv)
        """
        import json
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    content = chunk.get("content_with_weight", "")
                    f.write(f"=== Chunk {i+1} ===\n")
                    f.write(content)
                    f.write("\n\n")

        elif format.lower() == "csv":
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["chunk_id", "content", "tokens", "doc_name"])

                for i, chunk in enumerate(chunks):
                    content = chunk.get("content_with_weight", "")
                    tokens = num_tokens_from_string(content)
                    doc_name = chunk.get("docnm_kwd", "")
                    writer.writerow([i+1, content, tokens, doc_name])

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logging.info(f"Chunks exported to {output_path} in {format} format")

    @classmethod
    def get_supported_parsers(cls) -> List[str]:
        """Get list of supported parser types"""
        return list(cls.PARSER_FACTORY.keys())

    @classmethod
    def get_parser_info(cls, parser_type: str) -> Dict[str, Any]:
        """
        Get information about a specific parser

        Args:
            parser_type: Type of parser

        Returns:
            Dictionary containing parser information
        """
        parser_descriptions = {
            "general": "General purpose parser for common documents",
            "naive": "Simple text-based parser",
            "paper": "Academic paper parser with advanced layout recognition",
            "book": "Book parser with chapter and section detection",
            "presentation": "PowerPoint/presentation parser",
            "manual": "Technical manual parser",
            "laws": "Legal document parser",
            "qa": "Question-Answer document parser",
            "table": "Table-focused parser",
            "one": "Single-page document parser",
            "email": "Email document parser",
            "kg": "Knowledge graph parser",
            "tag": "Tag-based parser"
        }

        if parser_type not in cls.PARSER_FACTORY:
            raise ValueError(f"Unknown parser type: {parser_type}")

        return {
            "name": parser_type,
            "description": parser_descriptions.get(parser_type, "No description available"),
            "module": cls.PARSER_FACTORY[parser_type].__name__
        }
