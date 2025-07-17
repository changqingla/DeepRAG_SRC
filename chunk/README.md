# RAGFlow Document Chunker

A comprehensive document chunking solution based on RAGFlow's deep document understanding algorithms. This module provides intelligent document processing with OCR, layout recognition, table structure recognition, and sophisticated chunking strategies.

## Features

- **Complete RAGFlow Integration**: Reuses RAGFlow's entire document processing pipeline
- **Multiple Parser Types**: Support for 15+ specialized document parsers
- **Deep Document Understanding**: OCR, layout recognition, and table structure recognition
- **Intelligent Chunking**: Context-aware chunking with multiple strategies
- **Batch Processing**: Process multiple documents efficiently
- **Quality Analysis**: Built-in chunk quality metrics and analysis
- **Multiple Export Formats**: JSON, TXT, CSV output formats
- **Async Support**: Asynchronous processing capabilities
- **CLI Tool**: Command-line interface for easy usage

## Supported Document Types

| Document Type | Parser | Description |
|---------------|--------|-------------|
| PDF | `paper`, `book`, `manual`, `general` | Academic papers, books, manuals |
| DOCX/DOC | `book`, `manual`, `general` | Word documents |
| TXT/MD | `general`, `naive` | Plain text and markdown |
| XLSX/XLS/CSV | `table` | Spreadsheets and tables |
| PPTX/PPT | `presentation` | PowerPoint presentations |
| Images | `picture` | JPG, PNG, BMP, TIFF with OCR |
| Audio | `audio` | MP3, WAV, M4A transcripts |
| HTML | `general` | Web pages |
| Email | `email` | Email documents |
| Resume | `resume` | CV/Resume documents |
| Legal | `laws` | Legal documents |
| Q&A | `qa` | Question-answer documents |

## Installation

1. Ensure you have RAGFlow installed and configured
2. Copy the `dev` directory to your RAGFlow installation
3. Install any additional dependencies if needed

```bash
# Navigate to RAGFlow root directory
cd /path/to/ragflow

# The dev directory should be placed here
# ragflow/dev/
```

## Quick Start

### Basic Usage

```python
from dev.document_chunker import DocumentChunker

# Initialize chunker
chunker = DocumentChunker(
    parser_type="general",
    chunk_token_num=256,
    language="Chinese"
)

# Chunk a document
chunks = chunker.chunk_document("document.pdf")

# Get statistics
stats = chunker.get_chunk_statistics(chunks)
print(f"Generated {stats['total_chunks']} chunks")
```

### Command Line Usage

```bash
# Basic chunking
python dev/chunk_cli.py document.pdf --parser paper --output chunks.json

# Batch processing
python dev/chunk_cli.py --batch ./documents --format txt

# Get parser recommendations
python dev/chunk_cli.py document.pdf --recommend-parser

# Show statistics and preview
python dev/chunk_cli.py document.pdf --stats --preview
```

## Advanced Usage

### Custom Configuration

```python
from dev.chunker_utils import ChunkingConfig
from dev.document_chunker import DocumentChunker

# Create custom configuration
config = ChunkingConfig(
    parser_type="paper",
    chunk_token_num=512,
    language="English",
    delimiter="\n.!?",
    layout_recognize="DeepDOC"
)

# Initialize chunker with config
chunker = DocumentChunker(**config.to_dict())
```

### Batch Processing

```python
# Process multiple files
file_paths = ["doc1.pdf", "doc2.docx", "doc3.txt"]
results = chunker.chunk_batch(file_paths)

for file_path, chunks in results.items():
    print(f"{file_path}: {len(chunks)} chunks")
```

### Asynchronous Processing

```python
import asyncio

async def process_async():
    chunks = await chunker.chunk_document_async("large_document.pdf")
    return chunks

# Run async processing
chunks = asyncio.run(process_async())
```

## Parser Types and Use Cases

### Academic Papers (`paper`)
- Advanced layout recognition for academic documents
- Table and figure extraction
- Reference and citation handling
- Optimal for research papers and scientific documents

### Books (`book`)
- Chapter and section detection
- Hierarchical content organization
- Bullet point and list handling
- Best for books, manuals, and structured documents

### Presentations (`presentation`)
- Slide-based content extraction
- Image and text integration
- Optimal for PowerPoint and presentation files

### Tables (`table`)
- Specialized table structure recognition
- Cell content extraction and organization
- Perfect for spreadsheets and tabular data

### Pictures (`picture`)
- OCR for image-based documents
- Text extraction from images
- Support for various image formats

### General (`general`)
- Universal parser for common documents
- Balanced approach for mixed content
- Good default choice for most documents

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parser_type` | str | "general" | Type of parser to use |
| `chunk_token_num` | int | 256 | Maximum tokens per chunk |
| `delimiter` | str | "\n。；！？" | Text delimiters for chunking |
| `language` | str | "Chinese" | Document language |
| `layout_recognize` | str | "DeepDOC" | Layout recognition method |
| `zoomin` | int | 3 | Zoom factor for OCR |
| `from_page` | int | 0 | Starting page number |
| `to_page` | int | 100000 | Ending page number |

## Output Format

Each chunk is a dictionary containing:

```python
{
    "content_with_weight": "Chunk text content",
    "docnm_kwd": "document_name.pdf",
    "title_tks": ["tokenized", "title"],
    "title_sm_tks": ["fine", "grained", "tokens"],
    "content_ltks": ["content", "tokens"],
    "content_sm_ltks": ["fine", "grained", "content", "tokens"],
    "image": "base64_encoded_image",  # if applicable
    "positions": [[page, x0, y0, x1, y1]],  # position info
    # Additional metadata...
}
```

## Quality Analysis

The chunker provides built-in quality analysis:

```python
from dev.chunker_utils import ChunkAnalyzer

# Analyze chunk quality
quality_metrics = ChunkAnalyzer.analyze_chunk_quality(chunks)
print(f"Quality score: {quality_metrics['quality_score']}/100")
```

Quality metrics include:
- Total chunks and tokens
- Average chunk size
- Length variance
- Short/long chunk ratios
- Overall quality score

## Export Formats

### JSON Format
Complete chunk data with all metadata

### TXT Format
Plain text content with chunk separators

### CSV Format
Tabular format with chunk ID, content, tokens, and document name

## Command Line Interface

The CLI tool provides comprehensive options:

```bash
# Show help
python dev/chunk_cli.py --help

# Process with custom settings
python dev/chunk_cli.py document.pdf \
    --parser paper \
    --tokens 512 \
    --language English \
    --output results.json \
    --stats \
    --preview

# Batch processing with configuration
python dev/chunk_cli.py --batch ./docs \
    --config my_config.json \
    --format csv
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic document chunking
- Advanced configurations
- File type detection
- Batch processing
- Quality analysis
- Configuration management
- Export formats

## Error Handling

The chunker includes robust error handling:
- File not found errors
- Unsupported file types
- Processing failures
- Configuration validation

## Performance Considerations

- **Memory Usage**: Large documents may require significant memory
- **Processing Time**: Complex documents (PDFs with images) take longer
- **Batch Processing**: Use for multiple files to amortize initialization costs
- **Async Processing**: Use for I/O-bound operations

## Integration with RAGFlow

This chunker is designed to integrate seamlessly with RAGFlow:
- Uses the same parser factory as RAGFlow's task executor
- Compatible with RAGFlow's document processing pipeline
- Maintains consistency with RAGFlow's chunk format
- Can be used as a preprocessing step for RAG workflows

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure RAGFlow is properly installed and in Python path
2. **File Not Found**: Check file paths and permissions
3. **Parser Errors**: Verify parser type is supported for file type
4. **Memory Issues**: Reduce chunk size or process files individually

### Logging

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This module follows RAGFlow's development practices:
- Maintain compatibility with RAGFlow's core algorithms
- Add comprehensive tests for new features
- Follow existing code style and patterns
- Document all public APIs

## License

Apache 2.0 License - Same as RAGFlow

## Support

For issues and questions:
1. Check the examples and documentation
2. Review RAGFlow's main documentation
3. Submit issues with detailed error information and sample files
