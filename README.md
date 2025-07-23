# DeepRAG_SRC

<div align="center">

![DeepRAG Logo](https://img.shields.io/badge/DeepRAG-Document%20Intelligence-blue?style=for-the-badge)

**基于深度学习的智能文档处理与检索系统**

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-7.x%2F8.x-orange.svg)](https://www.elastic.co/)

[English](./README_EN.md) | 简体中文

</div>

## 🚀 项目概述

DeepRAG_SRC 是一个基于深度学习的智能文档处理与检索系统，提供从文档解析、分块、向量化到智能检索的完整解决方案。该项目基于 RAGFlow 的核心算法，去除了租户系统限制，提供更加灵活和高效的文档处理能力。

### ✨ 核心特性

- **🧠 深度文档理解**: 集成 DeepDOC 算法，支持 OCR、版面识别、表格结构识别
- **📄 多格式支持**: 支持 PDF、Word、Excel、PowerPoint、Markdown、HTML 等多种文档格式
- **🔧 智能分块**: 基于语义的智能分块策略，保持内容完整性和上下文连贯性
- **🎯 高质量向量化**: 支持 30+ 种嵌入模型，包括 VLLM、OpenAI、LocalAI 等
- **🔍 混合检索**: 结合文本搜索和向量搜索，支持智能重排序
- **⚡ 高性能存储**: 基于 Elasticsearch 的高效向量存储和检索
- **🌐 API 服务**: 提供完整的 HTTP API 接口，支持微服务架构
- **🖥️ 命令行工具**: 丰富的命令行工具，方便批量处理和测试

## 📁 项目架构

```
DeepRAG_SRC/
├── 📚 chunk/                    # 文档分块模块
│   ├── document_chunker.py      # 核心分块器
│   ├── chunker_utils.py         # 工具函数
│   └── chunk_cli.py             # 命令行工具
├── 🎯 embedding/                # 向量化模块
│   ├── chunk_embedder.py        # 分块嵌入器
│   ├── embedding_utils.py       # 嵌入工具
│   └── embed_cli.py             # 命令行工具
├── 💾 embed_store/              # 向量存储模块
│   ├── es_connection.py         # Elasticsearch 连接
│   └── simple_store.py          # 简单存储工具
├── 🔍 recall/                   # 召回检索模块
│   ├── deeprag_pure_retriever.py # 纯净召回器
│   ├── es_adapter.py            # ES 适配器
│   └── recall_cli.py            # 命令行工具
├── 🌐 api/                      # API 服务模块
│   ├── chunk_server.py          # 分块服务 (端口 8089)
│   └── embedding_server.py      # 向量化服务 (端口 8090)
├── 👁️ deepdoc/                  # 深度文档理解
│   ├── vision/                  # 视觉处理模块
│   └── parser/                  # 文档解析器
└── 🧠 rag/                      # 核心 RAG 算法
    ├── llm/                     # 大语言模型接口
    ├── nlp/                     # 自然语言处理
    ├── app/                     # 应用层解析器
    └── utils/                   # 工具函数
```

## 🎯 核心模块

### 1. 📚 文档分块模块 (chunk/)

基于 DeepRAG 的深度文档理解算法，提供智能文档分块功能。

**主要特性:**
- 支持 10 种专业解析器 (general, paper, book, presentation, manual, laws, qa, table, one, email)
- 智能版面识别和内容提取
- 基于语义的分块策略
- 支持异步和批量处理

**快速开始:**
```python
from chunk.document_chunker import DocumentChunker

chunker = DocumentChunker(parser_type="paper", chunk_token_num=512)
chunks = chunker.chunk_document("research_paper.pdf")
```

**命令行使用:**
```bash
python chunk/chunk_cli.py document.pdf --parser paper --output chunks.json
```

### 2. 🎯 向量化模块 (embedding/)

提供高质量的文档分块向量化功能，支持多种嵌入模型。

**支持的模型:**
- **VLLM**: 高性能推理服务
- **OpenAI**: GPT 系列嵌入模型
- **LocalAI**: 本地 AI 服务
- **通义千问**: 阿里云嵌入模型
- **智谱AI**: 清华智谱嵌入模型
- **BAAI**: 北京智源嵌入模型

**快速开始:**
```python
from embedding.chunk_embedder import ChunkEmbedder, EmbeddingConfig

config = EmbeddingConfig(
    model_factory="VLLM",
    model_name="bge-m3",
    base_url="http://localhost:8002/v1"
)
embedder = ChunkEmbedder(config)
token_count, vector_size = embedder.embed_chunks_sync(chunks)
```

**命令行使用:**
```bash
python embedding/embed_cli.py chunks.json \
  --model-factory VLLM \
  --model-name bge-m3 \
  --model-base-url http://localhost:8002/v1
```

### 3. 💾 向量存储模块 (embed_store/)

基于 Elasticsearch 的高效向量存储和管理。

**主要特性:**
- 自动检测和适配 IK 分词器
- 支持多种日期格式
- 高效的批量存储
- 完整的索引管理

**快速开始:**
```python
from embed_store.es_connection import ESConnection

es_conn = ESConnection(es_config={"hosts": "http://localhost:9200"})
es_conn.create_index("my_vectors", vector_size=1024)
es_conn.store_chunks(embedded_chunks, "my_vectors")
```

**命令行使用:**
```bash
python embed_store/simple_store.py chunks_embedded.json \
  --es-host http://localhost:9200 \
  --index my_vectors
```

### 4. 🔍 召回检索模块 (recall/)

基于 DeepRAG 原有算法的纯净召回系统，提供高质量的文档检索。

**主要特性:**
- 混合搜索 (文本 + 向量)
- 智能重排序
- 多种重排序模型支持
- 降级策略保证稳定性

**快速开始:**
```python
from recall.deeprag_pure_retriever import deepragPureRetriever, deepragRetrievalConfig

config = deepragRetrievalConfig(
    index_names=["my_index"],
    similarity_threshold=0.2,
    vector_similarity_weight=0.7
)
retriever = deepragPureRetriever(config)
result = retriever.retrieval(question="什么是人工智能？", embd_mdl=embedding_model)
```

**命令行使用:**
```bash
python recall/recall_cli.py "人工智能" \
  --indices my_index \
  --model-factory VLLM \
  --model-name bge-m3 \
  --model-base-url http://localhost:8002/v1 \
  --rerank-factory "OpenAI-API-Compatible" \
  --rerank-model-name bge-reranker-v2-m3 \
  --rerank-base-url http://localhost:8001/v1
```

### 5. 🌐 API 服务模块 (api/)

提供完整的 HTTP API 接口，支持微服务架构。

**服务列表:**
- **文档分块服务** (端口 8089): 提供文档分块 API
- **向量化服务** (端口 8090): 提供向量化和一站式处理 API

**启动服务:**
```bash
# 启动分块服务
python api/chunk_server.py --host 0.0.0.0 --port 8089

# 启动向量化服务
python api/embedding_server.py --host 0.0.0.0 --port 8090
```

**API 使用示例:**
```bash
# 文档分块
curl -X POST "http://localhost:8089/chunk" \
  -F "file=@document.pdf" \
  -F "parser_type=general"

# 一站式处理 (分块 + 向量化)
curl -X POST "http://localhost:8090/process" \
  -F "file=@document.pdf" \
  -F "model_factory=VLLM" \
  -F "model_name=bge-m3" \
  -F "base_url=http://localhost:8002/v1"
```

### 6. 👁️ 深度文档理解 (deepdoc/)

基于深度学习的文档视觉理解和解析。

**主要功能:**
- **OCR**: 光学字符识别
- **版面识别**: 智能版面分析
- **表格结构识别**: 复杂表格解析
- **图像处理**: 图像内容提取

**测试命令:**
```bash
# OCR 测试
python deepdoc/vision/t_ocr.py --inputs document.pdf --output_dir ./ocr_outputs

# 版面识别测试
python deepdoc/vision/t_recognizer.py --inputs document.pdf --mode layout --output_dir ./layout_outputs

# 表格结构识别测试
python deepdoc/vision/t_recognizer.py --inputs document.pdf --mode tsr --output_dir ./tsr_outputs
```

## 🛠️ 安装和配置

### 系统要求

- **Python**: 3.8+
- **Elasticsearch**: 7.x/8.x
- **内存**: 建议 8GB+
- **存储**: 建议 50GB+

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd DeepRAG_SRC
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置 Elasticsearch**
```bash
# 启动 Elasticsearch
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  elasticsearch:8.11.0
```

4. **配置向量化模型服务** (可选)
```bash
# 启动 VLLM 服务
python -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --port 8002
```

5. **配置重排序模型服务** (可选)
```bash
# 启动重排序服务
python -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-reranker-v2-m3 \
  --port 8001
```

### 环境变量配置

```bash
# Elasticsearch 配置
export ES_HOST=http://localhost:9200
export ES_TIMEOUT=600

# 模型服务配置
export EMBEDDING_SERVICE_URL=http://localhost:8002/v1
export RERANK_SERVICE_URL=http://localhost:8001/v1

# HuggingFace 镜像 (可选)
export HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 快速开始

### 完整的文档处理流程

```bash
# 1. 文档分块
python chunk/chunk_cli.py document.pdf --parser general --output chunks.json

# 2. 向量化
python embedding/embed_cli.py chunks.json \
  --model-factory VLLM \
  --model-name bge-m3 \
  --model-base-url http://localhost:8002/v1 \
  --output chunks_embedded.json

# 3. 存储到 Elasticsearch
python embed_store/simple_store.py chunks_embedded.json \
  --es-host http://localhost:9200 \
  --index my_vectors

# 4. 检索测试
python recall/recall_cli.py "什么是人工智能？" \
  --indices my_vectors \
  --model-factory VLLM \
  --model-name bge-m3 \
  --model-base-url http://localhost:8002/v1
```

### 使用 API 服务

```bash
# 启动服务
python api/chunk_server.py --port 8089 &
python api/embedding_server.py --port 8090 &

# 一站式处理
curl -X POST "http://localhost:8090/process" \
  -F "file=@document.pdf" \
  -F "model_factory=VLLM" \
  -F "model_name=bge-m3" \
  -F "base_url=http://localhost:8002/v1" \
  -F "chunk_token_num=512"
```


## 🔧 配置指南

### 文档分块配置

```json
{
  "parser_type": "general",
  "chunk_token_num": 256,
  "delimiter": "\n。；！？",
  "language": "Chinese",
  "layout_recognize": "DeepDOC",
  "zoomin": 3
}
```

### 向量化配置

```json
{
  "model_factory": "VLLM",
  "model_name": "bge-m3",
  "base_url": "http://localhost:8002/v1",
  "batch_size": 16,
  "filename_embd_weight": 0.1
}
```

### 检索配置

```json
{
  "similarity_threshold": 0.2,
  "vector_similarity_weight": 0.7,
  "top_k": 10,
  "rerank_enabled": true
}
```

## 📚 详细文档

- [文档分块模块](./chunk/README.md)
- [向量化模块](./embedding/README.md)
- [召回检索模块](./recall/README.md)
- [API 服务文档](./api/README.md)
- [DeepDOC 文档](./deepdoc/README.md)
