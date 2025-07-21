# DeepRAG 文档分块模块

## 📖 概述

DeepRAG 文档分块模块基于 DeepRAG 的深度文档理解算法，提供完整的文档分块功能。该模块复用了整个 DeepRAG 处理流水线，包括 OCR、版面识别、表格结构识别和智能分块策略，为文档处理提供高质量的分块结果。

## ✨ 主要特性

- **🔧 多格式支持**: 支持 PDF、Word、Excel、PowerPoint、Markdown、TXT、HTML 等多种文档格式
- **🧠 智能解析**: 集成 DeepRAG 的深度文档理解算法，支持 10 种专业解析器
- **📊 版面识别**: 智能版面识别和内容提取，支持表格和图像的结构化处理
- **⚡ 高效分块**: 基于语义的智能分块策略，保持内容完整性和上下文连贯性
- **🔄 异步处理**: 支持异步和批量处理，提高处理效率
- **🎯 灵活配置**: 丰富的配置选项，适应不同文档类型和处理需求
- **📈 质量分析**: 内置分块质量分析和统计功能
- **🖥️ 命令行工具**: 完整的命令行界面，方便批量操作

## 📁 模块结构

```
chunk/
├── README.md                    # 本文档
├── __init__.py                  # 模块初始化文件
├── document_chunker.py          # 核心文档分块器类
├── chunker_utils.py             # 工具函数和辅助类
├── chunk_cli.py                 # 命令行工具
├── config_examples.json         # 配置示例文件
└── markdown_chunks.json         # 示例分块结果
```

## 🚀 快速开始

### 1. 基本使用

```python
from document_chunker import DocumentChunker
from chunker_utils import ChunkingConfig

# 创建分块器
chunker = DocumentChunker(
    parser_type="general",
    chunk_token_num=256,
    language="Chinese"
)

# 分块文档
chunks = chunker.chunk_document("document.pdf")
print(f"生成了 {len(chunks)} 个分块")

# 查看分块内容
for i, chunk in enumerate(chunks[:3]):
    print(f"分块 {i+1}: {chunk['content_with_weight'][:100]}...")
```

### 2. 使用配置类

```python
from chunker_utils import ChunkingConfig

# 创建配置
config = ChunkingConfig(
    parser_type="paper",
    chunk_token_num=512,
    delimiter="\n.!?",
    language="English"
)

# 使用配置创建分块器
chunker = DocumentChunker(**config.to_dict())
chunks = chunker.chunk_document("research_paper.pdf")
```

### 3. 异步处理

```python
import asyncio

async def process_document():
    chunker = DocumentChunker()
    chunks = await chunker.chunk_document_async("document.pdf")
    return chunks

# 运行异步处理
chunks = asyncio.run(process_document())
```

### 4. 批量处理

```python
# 批量处理多个文档
file_paths = ["doc1.pdf", "doc2.docx", "doc3.pptx"]
results = chunker.chunk_batch(file_paths)

for file_path, chunks in results.items():
    print(f"{file_path}: {len(chunks)} 个分块")
```

## 🔧 支持的解析器

| 解析器类型 | 适用场景 | 特点 |
|-----------|---------|------|
| **general** | 通用文档 | 适用于大多数文档类型 |
| **paper** | 学术论文 | 优化论文结构识别 |
| **book** | 书籍文档 | 适合长文档和章节结构 |
| **presentation** | 演示文稿 | 针对 PPT 等演示文档 |
| **manual** | 技术手册 | 适合技术文档和说明书 |
| **laws** | 法律文档 | 专门处理法律条文 |
| **qa** | 问答文档 | 优化问答格式识别 |
| **table** | 表格数据 | 专门处理表格内容 |
| **one** | 单页文档 | 适合单页或简短文档 |
| **email** | 邮件文档 | 针对邮件格式优化 |

## 📋 支持的文件格式

| 格式类型 | 扩展名 | 说明 |
|---------|--------|------|
| **PDF** | `.pdf` | 支持文本和图像 PDF |
| **Word** | `.docx`, `.doc` | Microsoft Word 文档 |
| **Excel** | `.xlsx`, `.xls` | Excel 表格文件 |
| **PowerPoint** | `.pptx`, `.ppt` | PowerPoint 演示文稿 |
| **文本** | `.txt`, `.md` | 纯文本和 Markdown |
| **网页** | `.html`, `.htm` | HTML 网页文件 |

## 🛠️ 命令行使用

### 基本命令

```bash
# 分块单个文档
python chunk_cli.py document.pdf --parser paper --output chunks.json

# 批量处理目录
python chunk_cli.py --batch ./documents --parser book --format txt

# 使用自定义参数
python chunk_cli.py document.docx --tokens 512 --language English

# 获取解析器推荐
python chunk_cli.py document.pdf --recommend-parser

# 显示分块预览和统计
python chunk_cli.py document.pdf --preview --stats
```

### 高级功能

```bash
# 使用配置文件
python chunk_cli.py document.pdf --config config.json

# 保存配置
python chunk_cli.py document.pdf --save-config my_config.json

# 详细日志输出
python chunk_cli.py document.pdf --verbose

# 自定义分块参数
python chunk_cli.py document.pdf \
    --parser paper \
    --tokens 512 \
    --delimiter "\n.!?" \
    --language English \
    --from-page 1 \
    --to-page 10
```

## ⚙️ 配置参数

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `parser_type` | str | "general" | 解析器类型 |
| `chunk_token_num` | int | 256 | 每个分块的最大 token 数 |
| `delimiter` | str | "\n。；！？" | 文本分割符 |
| `language` | str | "Chinese" | 文档语言 |
| `layout_recognize` | str | "DeepDOC" | 版面识别方法 |
| `zoomin` | int | 3 | OCR 缩放因子 |
| `from_page` | int | 0 | 起始页码 |
| `to_page` | int | 100000 | 结束页码 |

### 配置示例

查看 `config_examples.json` 文件获取不同场景的配置示例：

```json
{
  "academic_paper": {
    "parser_type": "paper",
    "chunk_token_num": 512,
    "delimiter": "\n.!?",
    "language": "English"
  },
  "chinese_book": {
    "parser_type": "book",
    "chunk_token_num": 256,
    "delimiter": "\n。；！？",
    "language": "Chinese"
  }
}
```

## 📊 分块质量分析

### 统计信息

```python
# 获取分块统计
stats = chunker.get_chunk_statistics(chunks)
print(f"总分块数: {stats['total_chunks']}")
print(f"平均长度: {stats['avg_length']}")
print(f"token 分布: {stats['token_distribution']}")
```

### 质量分析

```python
from chunker_utils import ChunkAnalyzer

# 分析分块质量
analyzer = ChunkAnalyzer()
quality_report = analyzer.analyze_chunk_quality(chunks)

print(f"质量评分: {quality_report['quality_score']}")
print(f"优化建议: {quality_report['recommendations']}")
```

## 🔍 工具类功能

### 文件类型检测

```python
from chunker_utils import FileTypeDetector

# 检测文件类型
file_type = FileTypeDetector.detect_file_type("document.pdf")
print(f"文件类型: {file_type}")

# 获取解析器推荐
recommendations = FileTypeDetector.recommend_parser("document.pdf")
print(f"推荐解析器: {recommendations}")
```

### 结果格式化

```python
from chunker_utils import ResultFormatter

# 格式化显示
formatted = ResultFormatter.format_chunks_for_display(chunks)
for chunk in formatted[:3]:
    print(f"分块 {chunk['chunk_id']}: {chunk['content_preview']}")

# 生成报告
report = ResultFormatter.create_summary_report(chunks, processing_time=2.5)
print(report)
```

## 🚨 错误处理

### 常见错误

1. **文件不存在**: 检查文件路径是否正确
2. **格式不支持**: 确认文件格式在支持列表中
3. **内存不足**: 减少 `chunk_token_num` 或分批处理
4. **OCR 失败**: 调整 `zoomin` 参数或检查图像质量

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 验证参数
validation_result = chunker.validate_parameters()
if not validation_result['valid']:
    print(f"参数错误: {validation_result['errors']}")

# 使用进度回调
def progress_callback(progress, msg):
    print(f"进度: {progress:.1%} - {msg}")

chunker = DocumentChunker(progress_callback=progress_callback)
```

## 🎯 最佳实践

### 1. 选择合适的解析器

- **学术论文**: 使用 `paper` 解析器
- **技术文档**: 使用 `manual` 解析器
- **表格数据**: 使用 `table` 解析器
- **通用文档**: 使用 `general` 解析器

### 2. 优化分块大小

- **短文档**: 128-256 tokens
- **长文档**: 256-512 tokens
- **技术文档**: 384-512 tokens
- **对话数据**: 128-256 tokens

### 3. 语言设置

- 中文文档使用 `language="Chinese"`
- 英文文档使用 `language="English"`
- 根据主要语言选择合适的分隔符

### 4. 性能优化

- 大文档使用异步处理
- 批量处理多个文档
- 合理设置页面范围
- 根据需要调整 OCR 参数

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 📄 许可证

Apache 2.0 License