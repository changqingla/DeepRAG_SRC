#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow分块存储命令行工具

本脚本提供命令行界面，用于使用RAGFlow存储逻辑将已向量化的分块
存储到Elasticsearch中。

用法:
    python store_cli.py embedded_chunks.json [选项]

作者: RAGFlow开发团队
许可证: Apache 2.0
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict, Any

# Import our modules
try:
    from chunk_store import ChunkStore, ChunkStoreConfig
    from store_utils import (
        StorageConfigManager, ChunkValidator, StorageAnalyzer,
        StorageResult, StorageExporter
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the embed_store directory")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RAGFlow Chunk Store - Store embedded chunks to Elasticsearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store chunks with default configuration
  python store_cli.py embedded_chunks.json
  
  # Store with custom index name
  python store_cli.py embedded_chunks.json --index-name my_documents
  
  # Store with tenant-style configuration
  python store_cli.py embedded_chunks.json --tenant-id user123
  
  # Validate chunks before storing
  python store_cli.py embedded_chunks.json --validate-only
  
  # Store with custom batch size
  python store_cli.py embedded_chunks.json --batch-size 8
        """
    )
    
    # Input options
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input JSON file containing embedded chunks'
    )
    
    # Storage configuration
    parser.add_argument(
        '--index-name',
        help='Elasticsearch index name (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--tenant-id',
        help='Tenant ID for RAGFlow-style index naming'
    )
    
    parser.add_argument(
        '--kb-id',
        help='Knowledge base ID (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--doc-id',
        help='Document ID (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for bulk operations (default: 4)'
    )
    
    parser.add_argument(
        '--no-auto-create',
        action='store_true',
        help='Do not auto-create index if it does not exist'
    )
    
    # Validation options
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate chunks without storing'
    )
    
    parser.add_argument(
        '--fix-chunks',
        action='store_true',
        help='Attempt to fix common issues in chunks'
    )
    
    # Output options
    parser.add_argument(
        '--output-report',
        help='Output file for storage report'
    )
    
    parser.add_argument(
        '--output-config',
        help='Output file for storage configuration'
    )
    
    parser.add_argument(
        '--export-mapping',
        help='Export index mapping information to file'
    )
    
    parser.add_argument(
        '--export-sample',
        help='Export sample chunk structure to file'
    )
    
    # Configuration management
    parser.add_argument(
        '--save-config',
        help='Save storage configuration to file'
    )
    
    parser.add_argument(
        '--load-config',
        help='Load storage configuration from file'
    )
    
    # Index management
    parser.add_argument(
        '--delete-index',
        action='store_true',
        help='Delete the index after operation (use with caution!)'
    )
    
    parser.add_argument(
        '--show-index-info',
        action='store_true',
        help='Show index information'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list):
            raise ValueError("Input file must contain a list of chunks")
        
        logging.info(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
        
    except Exception as e:
        logging.error(f"Failed to load chunks from {file_path}: {e}")
        sys.exit(1)


def create_storage_config(args) -> ChunkStoreConfig:
    """Create storage configuration from arguments"""
    if args.load_config:
        return StorageConfigManager.load_config(Path(args.load_config))
    
    config_kwargs = {
        'batch_size': args.batch_size,
        'auto_create_index': not args.no_auto_create
    }
    
    if args.kb_id:
        config_kwargs['kb_id'] = args.kb_id
    
    if args.doc_id:
        config_kwargs['doc_id'] = args.doc_id
    
    if args.tenant_id:
        return StorageConfigManager.create_tenant_config(args.tenant_id, **config_kwargs)
    elif args.index_name:
        return StorageConfigManager.create_default_config(args.index_name, **config_kwargs)
    else:
        return StorageConfigManager.create_default_config(**config_kwargs)


def validate_chunks(chunks: List[Dict[str, Any]], fix_issues: bool = False) -> List[Dict[str, Any]]:
    """Validate and optionally fix chunks"""
    logging.info("Validating chunks...")
    
    validation_result = ChunkValidator.validate_chunks(chunks)
    
    print(f"\nChunk Validation Results:")
    print(f"Total chunks: {validation_result['total_chunks']}")
    print(f"Valid: {validation_result['valid']}")
    print(f"Vector fields found: {validation_result['vector_fields']}")
    
    if validation_result['warnings']:
        print(f"\nWarnings ({len(validation_result['warnings'])}):")
        for warning in validation_result['warnings'][:5]:
            print(f"  - {warning}")
        if len(validation_result['warnings']) > 5:
            print(f"  ... and {len(validation_result['warnings']) - 5} more warnings")
    
    if validation_result['errors']:
        print(f"\nErrors ({len(validation_result['errors'])}):")
        for error in validation_result['errors'][:5]:
            print(f"  - {error}")
        if len(validation_result['errors']) > 5:
            print(f"  ... and {len(validation_result['errors']) - 5} more errors")
        
        if not fix_issues:
            print("\nValidation failed. Use --fix-chunks to attempt automatic fixes.")
            sys.exit(1)
    
    if fix_issues and (validation_result['errors'] or validation_result['warnings']):
        logging.info("Attempting to fix chunk issues...")
        fixed_chunks = ChunkValidator.fix_chunks(chunks)
        
        # Re-validate
        revalidation = ChunkValidator.validate_chunks(fixed_chunks)
        if revalidation['valid']:
            print("✅ Chunks fixed successfully!")
            return fixed_chunks
        else:
            print("❌ Could not fix all issues automatically.")
            sys.exit(1)
    
    return chunks


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if input file is provided
    if not args.input_file:
        if args.show_index_info or args.delete_index:
            # These operations don't require input file
            pass
        else:
            logging.error("Input file is required")
            sys.exit(1)
    
    # Load chunks if input file provided
    chunks = []
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            logging.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        chunks = load_chunks(input_path)
        if not chunks:
            logging.error("No chunks found in input file")
            sys.exit(1)
    
    # Create storage configuration
    try:
        config = create_storage_config(args)
        logging.info(f"Storage configuration: index={config.index_name}, kb_id={config.kb_id}")
    except Exception as e:
        logging.error(f"Failed to create storage configuration: {e}")
        sys.exit(1)
    
    # Save configuration if requested
    if args.save_config:
        StorageConfigManager.save_config(config, Path(args.save_config))
        logging.info(f"Configuration saved to {args.save_config}")
    
    # Initialize chunk store
    try:
        store = ChunkStore(config)
    except Exception as e:
        logging.error(f"Failed to initialize chunk store: {e}")
        sys.exit(1)
    
    # Show index info if requested
    if args.show_index_info:
        index_info = store.get_index_info()
        print(f"\nIndex Information:")
        for key, value in index_info.items():
            print(f"  {key}: {value}")
        print()
    
    # Export operations
    if args.export_mapping:
        StorageExporter.export_index_mapping(config.index_name, Path(args.export_mapping))
        logging.info(f"Index mapping exported to {args.export_mapping}")
    
    if args.export_sample and chunks:
        StorageExporter.export_sample_chunk(chunks, Path(args.export_sample))
        logging.info(f"Sample chunk exported to {args.export_sample}")
    
    # Validate chunks
    if chunks:
        chunks = validate_chunks(chunks, args.fix_chunks)
        
        if args.validate_only:
            print("✅ Validation completed successfully!")
            return
    
    # Store chunks
    if chunks:
        logging.info("Starting chunk storage...")
        start_time = timer()
        
        try:
            stored_count, error_messages = store.store_chunks(chunks)
            processing_time = timer() - start_time
            
            # Create result object
            result = StorageResult(
                stored_count=stored_count,
                total_count=len(chunks),
                error_count=len(error_messages),
                error_messages=error_messages,
                processing_time=processing_time,
                index_info=store.get_index_info()
            )
            
            # Generate report
            report = StorageAnalyzer.create_storage_report(result)
            print(report)
            
            # Save report if requested
            if args.output_report:
                with open(args.output_report, 'w', encoding='utf-8') as f:
                    f.write(report)
                logging.info(f"Report saved to {args.output_report}")
            
            # Check if storage was successful
            if not result.is_successful:
                logging.error("Storage operation failed")
                sys.exit(1)
            
        except Exception as e:
            logging.error(f"Failed to store chunks: {e}")
            sys.exit(1)
    
    # Delete index if requested (use with caution!)
    if args.delete_index:
        if input("Are you sure you want to delete the index? (yes/no): ").lower() == 'yes':
            if store.delete_index():
                print("✅ Index deleted successfully")
            else:
                print("❌ Failed to delete index")
                sys.exit(1)
        else:
            print("Index deletion cancelled")
    
    logging.info("Operation completed successfully")


if __name__ == "__main__":
    main()
