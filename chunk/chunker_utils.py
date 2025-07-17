#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow Document Chunker Utilities

This module provides utility functions and helper classes for the RAGFlow-based
document chunker, including file type detection, configuration management,
and result analysis tools.

Author: RAGFlow Dev Team
License: Apache 2.0
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
    """Configuration class for document chunking"""
    parser_type: str = "general"
    chunk_token_num: int = 256
    delimiter: str = "\n。；！？"
    language: str = "Chinese"
    layout_recognize: str = "DeepDOC"
    zoomin: int = 3
    from_page: int = 0
    to_page: int = 100000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChunkingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ChunkingConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class FileTypeDetector:
    """Utility class for detecting file types and recommending parsers"""
    
    # File extension to parser type mapping
    EXTENSION_PARSER_MAP = {
        '.pdf': ['paper', 'book', 'manual', 'general'],
        '.docx': ['book', 'manual', 'general'],
        '.doc': ['book', 'manual', 'general'],
        '.txt': ['general', 'naive'],
        '.md': ['general', 'naive'],
        '.html': ['general'],
        '.htm': ['general'],
        '.xlsx': ['table'],
        '.xls': ['table'],
        '.csv': ['table'],
        '.pptx': ['presentation'],
        '.ppt': ['presentation'],
        '.jpg': ['picture'],
        '.jpeg': ['picture'],
        '.png': ['picture'],
        '.bmp': ['picture'],
        '.tiff': ['picture'],
        '.mp3': ['audio'],
        '.wav': ['audio'],
        '.m4a': ['audio'],
        '.json': ['general'],
        '.xml': ['general']
    }
    
    @classmethod
    def detect_file_type(cls, file_path: Union[str, Path]) -> str:
        """
        Detect file type based on extension and MIME type
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if extension in cls.EXTENSION_PARSER_MAP:
            return extension
        elif mime_type:
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
        
        return '.txt'  # Default fallback
    
    @classmethod
    def recommend_parser(cls, file_path: Union[str, Path]) -> List[str]:
        """
        Recommend parser types for a given file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of recommended parser types (ordered by preference)
        """
        file_type = cls.detect_file_type(file_path)
        return cls.EXTENSION_PARSER_MAP.get(file_type, ['general'])
    
    @classmethod
    def is_supported_file(cls, file_path: Union[str, Path]) -> bool:
        """
        Check if file type is supported
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported
        """
        file_type = cls.detect_file_type(file_path)
        return file_type in cls.EXTENSION_PARSER_MAP


class ChunkAnalyzer:
    """Utility class for analyzing chunking results"""
    
    @staticmethod
    def analyze_chunk_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary containing quality metrics
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Extract content lengths
        content_lengths = []
        token_counts = []
        
        for chunk in chunks:
            content = chunk.get("content_with_weight", "")
            if content:
                content_lengths.append(len(content))
                # Estimate token count (rough approximation)
                token_count = len(content.split())
                token_counts.append(token_count)
        
        if not content_lengths:
            return {"error": "No valid content found in chunks"}
        
        # Calculate statistics
        avg_length = sum(content_lengths) / len(content_lengths)
        avg_tokens = sum(token_counts) / len(token_counts)
        
        # Quality metrics
        length_variance = sum((x - avg_length) ** 2 for x in content_lengths) / len(content_lengths)
        token_variance = sum((x - avg_tokens) ** 2 for x in token_counts) / len(token_counts)
        
        # Check for very short or very long chunks
        short_chunks = sum(1 for length in content_lengths if length < 50)
        long_chunks = sum(1 for length in content_lengths if length > 2000)
        
        return {
            "total_chunks": len(chunks),
            "avg_content_length": avg_length,
            "avg_token_count": avg_tokens,
            "length_variance": length_variance,
            "token_variance": token_variance,
            "short_chunks_count": short_chunks,
            "long_chunks_count": long_chunks,
            "short_chunks_ratio": short_chunks / len(chunks),
            "long_chunks_ratio": long_chunks / len(chunks),
            "quality_score": ChunkAnalyzer._calculate_quality_score(
                short_chunks / len(chunks),
                long_chunks / len(chunks),
                length_variance / (avg_length ** 2) if avg_length > 0 else 0
            )
        }
    
    @staticmethod
    def _calculate_quality_score(short_ratio: float, long_ratio: float, 
                               normalized_variance: float) -> float:
        """
        Calculate a quality score for chunks (0-100)
        
        Args:
            short_ratio: Ratio of very short chunks
            long_ratio: Ratio of very long chunks
            normalized_variance: Normalized variance in chunk lengths
            
        Returns:
            Quality score (higher is better)
        """
        # Penalize high ratios of short/long chunks and high variance
        penalty = (short_ratio * 30) + (long_ratio * 20) + (normalized_variance * 50)
        score = max(0, 100 - penalty)
        return score
    
    @staticmethod
    def find_optimal_chunk_size(sample_text: str, 
                              target_chunks: int = 10) -> int:
        """
        Find optimal chunk size for a given text
        
        Args:
            sample_text: Sample text to analyze
            target_chunks: Target number of chunks
            
        Returns:
            Recommended chunk token size
        """
        # Estimate total tokens
        total_tokens = len(sample_text.split())
        
        # Calculate optimal chunk size
        optimal_size = total_tokens // target_chunks
        
        # Apply reasonable bounds
        optimal_size = max(128, min(optimal_size, 512))
        
        return optimal_size


class ResultFormatter:
    """Utility class for formatting chunking results"""
    
    @staticmethod
    def format_chunks_for_display(chunks: List[Dict[str, Any]], 
                                max_content_length: int = 200) -> List[Dict[str, Any]]:
        """
        Format chunks for display purposes
        
        Args:
            chunks: List of document chunks
            max_content_length: Maximum content length to display
            
        Returns:
            Formatted chunks for display
        """
        formatted = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content_with_weight", "")
            
            # Truncate content if too long
            if len(content) > max_content_length:
                display_content = content[:max_content_length] + "..."
            else:
                display_content = content
            
            formatted_chunk = {
                "chunk_id": i + 1,
                "content_preview": display_content,
                "content_length": len(content),
                "token_count": len(content.split()),
                "doc_name": chunk.get("docnm_kwd", "Unknown"),
                "has_image": "image" in chunk,
                "metadata": {k: v for k, v in chunk.items() 
                           if k not in ["content_with_weight", "image"]}
            }
            
            formatted.append(formatted_chunk)
        
        return formatted
    
    @staticmethod
    def create_summary_report(chunks: List[Dict[str, Any]], 
                            processing_time: float = None) -> str:
        """
        Create a summary report of chunking results
        
        Args:
            chunks: List of document chunks
            processing_time: Time taken for processing
            
        Returns:
            Formatted summary report
        """
        if not chunks:
            return "No chunks generated."
        
        # Basic statistics
        total_chunks = len(chunks)
        total_content = sum(len(chunk.get("content_with_weight", "")) for chunk in chunks)
        avg_chunk_size = total_content / total_chunks if total_chunks > 0 else 0
        
        # Quality analysis
        quality_metrics = ChunkAnalyzer.analyze_chunk_quality(chunks)
        
        report = f"""
Document Chunking Summary Report
================================

Basic Statistics:
- Total chunks generated: {total_chunks}
- Total content length: {total_content:,} characters
- Average chunk size: {avg_chunk_size:.1f} characters
- Quality score: {quality_metrics.get('quality_score', 0):.1f}/100

Quality Metrics:
- Short chunks (< 50 chars): {quality_metrics.get('short_chunks_count', 0)} ({quality_metrics.get('short_chunks_ratio', 0):.1%})
- Long chunks (> 2000 chars): {quality_metrics.get('long_chunks_count', 0)} ({quality_metrics.get('long_chunks_ratio', 0):.1%})
- Content length variance: {quality_metrics.get('length_variance', 0):.1f}
"""
        
        if processing_time:
            report += f"\nProcessing Time: {processing_time:.2f} seconds"
        
        return report
