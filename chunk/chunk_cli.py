#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow Document Chunker Command Line Interface

This script provides a command-line interface for the RAGFlow-based document
chunker, allowing users to process documents from the command line with
various options and configurations.

Usage:
    python chunk_cli.py input_file [options]
    python chunk_cli.py --batch input_directory [options]

Author: RAGFlow Dev Team
License: Apache 2.0
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from timeit import default_timer as timer

# Import our modules
from document_chunker import DocumentChunker
from chunker_utils import (
    ChunkingConfig, FileTypeDetector, ChunkAnalyzer, 
    ResultFormatter
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RAGFlow Document Chunker - Process documents using RAGFlow's deep understanding algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chunk a single PDF using paper parser
  python chunk_cli.py document.pdf --parser paper --output chunks.json
  
  # Batch process all PDFs in a directory
  python chunk_cli.py --batch ./documents --parser book --format txt
  
  # Use custom chunking parameters
  python chunk_cli.py document.docx --tokens 512 --language English
  
  # Get parser recommendations for a file
  python chunk_cli.py document.pdf --recommend-parser
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'input_file', 
        nargs='?',
        help='Input document file to process'
    )
    input_group.add_argument(
        '--batch', 
        metavar='DIRECTORY',
        help='Process all supported files in the specified directory'
    )
    
    # Parser options
    parser.add_argument(
        '--parser', 
        choices=DocumentChunker.get_supported_parsers(),
        default='general',
        help='Parser type to use (default: general)'
    )
    
    parser.add_argument(
        '--recommend-parser',
        action='store_true',
        help='Show recommended parsers for the input file and exit'
    )
    
    # Chunking parameters
    parser.add_argument(
        '--tokens',
        type=int,
        default=256,
        help='Maximum tokens per chunk (default: 256)'
    )
    
    parser.add_argument(
        '--delimiter',
        default="\n。；！？",
        help='Text delimiters for chunking (default: "\\n。；！？")'
    )
    
    parser.add_argument(
        '--language',
        choices=['Chinese', 'English'],
        default='Chinese',
        help='Document language (default: Chinese)'
    )
    
    parser.add_argument(
        '--from-page',
        type=int,
        default=0,
        help='Starting page number (default: 0)'
    )
    
    parser.add_argument(
        '--to-page',
        type=int,
        default=100000,
        help='Ending page number (default: 100000)'
    )
    
    parser.add_argument(
        '--zoomin',
        type=int,
        default=3,
        help='Zoom factor for OCR (default: 3)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: auto-generated based on input)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'txt', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics about the chunks'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview of generated chunks'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        help='Load configuration from JSON file'
    )
    
    parser.add_argument(
        '--save-config',
        help='Save current configuration to JSON file'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> ChunkingConfig:
    """Load configuration from file"""
    try:
        return ChunkingConfig.load_from_file(config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)


def save_config(config: ChunkingConfig, config_path: str):
    """Save configuration to file"""
    try:
        config.save_to_file(config_path)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration to {config_path}: {e}")


def get_output_path(input_path: Path, output_arg: str, format: str) -> Path:
    """Generate output file path"""
    if output_arg:
        return Path(output_arg)
    
    # Auto-generate output path
    base_name = input_path.stem
    extension = format
    return input_path.parent / f"{base_name}_chunks.{extension}"


def process_single_file(file_path: Path, chunker: DocumentChunker, args) -> List[Dict[str, Any]]:
    """Process a single file"""
    logging.info(f"Processing file: {file_path}")
    
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return []
    
    if not FileTypeDetector.is_supported_file(file_path):
        logging.warning(f"File type may not be supported: {file_path}")
    
    try:
        start_time = timer()
        chunks = chunker.chunk_document(file_path)
        processing_time = timer() - start_time
        
        logging.info(f"Successfully processed {file_path}")
        logging.info(f"Generated {len(chunks)} chunks in {processing_time:.2f}s")
        
        return chunks
        
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return []


def process_batch(directory: Path, chunker: DocumentChunker, args) -> Dict[str, List[Dict[str, Any]]]:
    """Process all supported files in a directory"""
    logging.info(f"Processing batch directory: {directory}")
    
    if not directory.exists() or not directory.is_dir():
        logging.error(f"Directory not found: {directory}")
        return {}
    
    # Find all supported files
    supported_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and FileTypeDetector.is_supported_file(file_path):
            supported_files.append(file_path)
    
    if not supported_files:
        logging.warning(f"No supported files found in {directory}")
        return {}
    
    logging.info(f"Found {len(supported_files)} supported files")
    
    # Process files
    results = {}
    for file_path in supported_files:
        chunks = process_single_file(file_path, chunker, args)
        results[str(file_path)] = chunks
    
    return results


def show_parser_recommendations(file_path: Path):
    """Show recommended parsers for a file"""
    recommendations = FileTypeDetector.recommend_parser(file_path)
    file_type = FileTypeDetector.detect_file_type(file_path)
    
    print(f"\nFile: {file_path}")
    print(f"Detected type: {file_type}")
    print(f"Recommended parsers (in order of preference):")
    
    for i, parser in enumerate(recommendations, 1):
        info = DocumentChunker.get_parser_info(parser)
        print(f"  {i}. {parser} - {info['description']}")
    
    print()


def show_chunk_preview(chunks: List[Dict[str, Any]], max_chunks: int = 5):
    """Show preview of chunks"""
    if not chunks:
        print("No chunks to preview.")
        return
    
    formatted_chunks = ResultFormatter.format_chunks_for_display(chunks)
    
    print(f"\nChunk Preview (showing first {min(max_chunks, len(chunks))} chunks):")
    print("=" * 60)
    
    for chunk in formatted_chunks[:max_chunks]:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Length: {chunk['content_length']} chars, {chunk['token_count']} tokens")
        print(f"  Content: {chunk['content_preview']}")
        if chunk['has_image']:
            print("  [Contains image]")
    
    if len(chunks) > max_chunks:
        print(f"\n... and {len(chunks) - max_chunks} more chunks")


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = ChunkingConfig(
            parser_type=args.parser,
            chunk_token_num=args.tokens,
            delimiter=args.delimiter,
            language=args.language,
            from_page=args.from_page,
            to_page=args.to_page,
            zoomin=args.zoomin
        )
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
    
    # Handle parser recommendations
    if args.recommend_parser:
        if args.input_file:
            show_parser_recommendations(Path(args.input_file))
        else:
            logging.error("--recommend-parser requires an input file")
        return
    
    # Initialize chunker
    try:
        chunker = DocumentChunker(**config.to_dict())
    except Exception as e:
        logging.error(f"Failed to initialize chunker: {e}")
        sys.exit(1)
    
    # Process files
    if args.batch:
        # Batch processing
        results = process_batch(Path(args.batch), chunker, args)
        
        # Save results for each file
        for file_path, chunks in results.items():
            if chunks:
                input_path = Path(file_path)
                output_path = get_output_path(input_path, None, args.format)
                chunker.export_chunks(chunks, output_path, args.format)
                
                if args.stats:
                    stats = chunker.get_chunk_statistics(chunks)
                    print(f"\nStatistics for {input_path.name}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
    
    else:
        # Single file processing
        input_path = Path(args.input_file)
        chunks = process_single_file(input_path, chunker, args)
        
        if chunks:
            # Save results
            output_path = get_output_path(input_path, args.output, args.format)
            chunker.export_chunks(chunks, output_path, args.format)
            
            # Show statistics
            if args.stats:
                stats = chunker.get_chunk_statistics(chunks)
                print(f"\nChunk Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            # Show preview
            if args.preview:
                show_chunk_preview(chunks)
            
            # Show summary
            processing_time = None  # We'd need to track this properly
            summary = ResultFormatter.create_summary_report(chunks, processing_time)
            print(summary)
    
    logging.info("Processing completed successfully")


if __name__ == "__main__":
    main()
