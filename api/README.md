# DeepRAG API 服务

DeepRAG 提供两个高性能的 HTTP API 服务：

- **文档分块服务** (`chunk_server.py`) - 端口 8089
- **向量化服务** (`embedding_server.py`) - 端口 8090

## 🚀 快速启动

```bash
# 启动文档分块服务
python api/chunk_server.py --host 0.0.0.0 --port 8089

# 启动向量化服务  
python api/embedding_server.py --host 0.0.0.0 --port 8090
```

## 📋 文档分块服务 API (端口 8089)

### 1. 服务状态

```bash
curl http://localhost:8089/
```

**响应:**
```json
{
  "service": "DeepRAG 文档分块服务",
  "version": "2.0.0",
  "status": "running",
  "supported_formats": [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".pptx", ".xlsx"],
  "max_file_size": "100MB",
  "max_concurrent_tasks": 50,
  "api_docs": "/docs"
}
```

### 2. 健康检查

```bash
curl http://localhost:8089/health
```

**响应:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "uptime": 3600.5,
  "current_concurrent_tasks": 2,
  "total_requests": 150
}
```

### 3. 单文档分块

```bash
curl -X POST "http://localhost:8089/chunk" \
  -F "file=@example.pdf" \
  -F "parser_type=auto" \
  -F "chunk_token_num=256" \
  -F "delimiter=\n。；！？" \
  -F "language=Chinese" \
  -F "layout_recognize=DeepDOC" \
  -F "zoomin=3" \
  -F "from_page=0" \
  -F "to_page=100000"
```

**响应:**
```json
{
  "success": true,
  "chunks": [
    {
      "content_with_weight": "这是第一个分块的内容...",
      "docnm_kwd": "example.pdf",
      "page_num_int": [1],
      "position_int": [[1, 0, 0, 0, 0]],
      "top_int": [0],
      "content_ltks": "这是 第一个 分块 的 内容",
      "content_sm_ltks": "这 是 第 一 个 分 块 的 内 容"
    },
    {
      "content_with_weight": "这是第二个分块的内容...",
      "docnm_kwd": "example.pdf",
      "page_num_int": [1],
      "position_int": [[2, 0, 0, 0, 0]],
      "top_int": [0],
      "content_ltks": "这是 第二个 分块 的 内容",
      "content_sm_ltks": "这 是 第 二 个 分 块 的 内 容"
    }
  ],
  "total_chunks": 2,
  "processing_time": 1.23,
  "file_size": 1048576,
  "parser_type": "general"
}
```

### 4. 批量文档分块

```bash
curl -X POST "http://localhost:8089/chunk/batch" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "files=@doc3.txt" \
  -F "parser_type=auto" \
  -F "chunk_token_num=256"
```

**响应:**
```json
{
  "success": true,
  "results": [
    {
      "filename": "doc1.pdf",
      "success": true,
      "chunks": [...],
      "total_chunks": 5,
      "processing_time": 2.1,
      "file_size": 2048576,
      "parser_type": "general"
    },
    {
      "filename": "doc2.docx",
      "success": true,
      "chunks": [...],
      "total_chunks": 3,
      "processing_time": 1.5,
      "file_size": 1024000,
      "parser_type": "general"
    },
    {
      "filename": "doc3.txt",
      "success": true,
      "chunks": [...],
      "total_chunks": 2,
      "processing_time": 0.8,
      "file_size": 512000,
      "parser_type": "general"
    }
  ],
  "total_files": 3,
  "successful_files": 3,
  "failed_files": 0,
  "total_processing_time": 4.4
}
```

### 5. 服务统计

```bash
curl http://localhost:8089/stats
```

**响应:**
```json
{
  "uptime": 7200.5,
  "total_requests": 250,
  "successful_requests": 240,
  "failed_requests": 10,
  "average_processing_time": 1.85,
  "current_concurrent_tasks": 0
}
```

### 6. 清理临时文件

```bash
curl -X POST "http://localhost:8089/admin/cleanup"
```

**响应:**
```json
{
  "message": "已清理 15 个临时文件",
  "cleaned_files": 15
}
```

## 🎯 向量化服务 API (端口 8090)

### 1. 服务状态

```bash
curl http://localhost:8090/
```

**响应:**
```json
{
  "service": "DeepRAG 向量化服务",
  "version": "1.0.0",
  "status": "running",
  "active_tasks": 2,
  "max_concurrent_tasks": 100
}
```

### 2. 健康检查

```bash
curl http://localhost:8090/health
```

**响应:**
```json
{
  "status": "healthy",
  "active_tasks": 0,
  "max_concurrent_tasks": 100
}
```

### 3. 分块向量化

```bash
curl -X POST "http://localhost:8090/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "content_with_weight": "这是第一个分块的内容...",
        "docnm_kwd": "example.pdf"
      },
      {
        "content_with_weight": "这是第二个分块的内容...",
        "docnm_kwd": "example.pdf"
      }
    ],
    "model_factory": "VLLM",
    "model_name": "bge-m3",
    "base_url": "http://localhost:8002/v1",
    "batch_size": 16,
    "filename_embd_weight": 0.1
  }'
```

**响应:**
```json
{
  "success": true,
  "message": "成功向量化 2 个分块",
  "chunks": [
    {
      "content_with_weight": "这是第一个分块的内容...",
      "docnm_kwd": "example.pdf",
      "q_1024_vec": [0.1, 0.2, 0.3, ..., 0.9]
    },
    {
      "content_with_weight": "这是第二个分块的内容...",
      "docnm_kwd": "example.pdf",
      "q_1024_vec": [0.2, 0.3, 0.4, ..., 0.8]
    }
  ],
  "stats": {
    "total_chunks": 2,
    "total_tokens": 50,
    "vector_dimension": 1024,
    "model_factory": "VLLM",
    "model_name": "bge-m3"
  },
  "processing_time": 0.85
}
```

### 4. 文档处理 (分块 + 向量化)

```bash
curl -X POST "http://localhost:8090/process" \
  -F "file=@example.pdf" \
  -F "model_factory=VLLM" \
  -F "model_name=bge-m3" \
  -F "base_url=http://localhost:8002/v1" \
  -F "parser_type=general" \
  -F "chunk_token_num=256" \
  -F "delimiter=\n。；！？" \
  -F "language=Chinese" \
  -F "layout_recognize=DeepDOC" \
  -F "batch_size=16" \
  -F "filename_embd_weight=0.1"
```

**响应:**
```json
{
  "success": true,
  "message": "成功处理文档 example.pdf，生成 3 个向量化分块",
  "chunks": [
    {
      "content_with_weight": "这是第一个分块的内容...",
      "docnm_kwd": "example.pdf",
      "q_1024_vec": [0.1, 0.2, 0.3, ..., 0.9],
      "content_ltks": "这是 第一个 分块 的 内容",
      "page_num_int": [1]
    },
    {
      "content_with_weight": "这是第二个分块的内容...",
      "docnm_kwd": "example.pdf",
      "q_1024_vec": [0.2, 0.3, 0.4, ..., 0.8],
      "content_ltks": "这是 第二个 分块 的 内容",
      "page_num_int": [1]
    },
    {
      "content_with_weight": "这是第三个分块的内容...",
      "docnm_kwd": "example.pdf",
      "q_1024_vec": [0.3, 0.4, 0.5, ..., 0.7],
      "content_ltks": "这是 第三个 分块 的 内容",
      "page_num_int": [2]
    }
  ],
  "chunk_stats": {
    "total_chunks": 3,
    "chunk_time": 1.2,
    "parser_type": "general",
    "chunk_token_num": 256
  },
  "embedding_stats": {
    "total_chunks": 3,
    "total_tokens": 75,
    "vector_dimension": 1024,
    "model_factory": "VLLM",
    "model_name": "bge-m3",
    "embedding_time": 0.95
  },
  "total_processing_time": 2.15,
  "chunk_time": 1.2,
  "embedding_time": 0.95
}
```

### 5. 列出支持的模型

```bash
curl http://localhost:8090/models
```

**响应:**
```json
{
  "supported_models": {
    "BAAI": {
      "factory_name": "BAAI",
      "description": "BAAI 嵌入模型"
    },
    "OpenAI": {
      "factory_name": "OpenAI",
      "description": "OpenAI 嵌入模型"
    },
    "VLLM": {
      "factory_name": "VLLM",
      "description": "VLLM 嵌入模型"
    },
    "LocalAI": {
      "factory_name": "LocalAI",
      "description": "LocalAI 嵌入模型"
    }
  },
  "total_count": 4
}
```

## 🔧 常用参数说明

### 分块参数
- `parser_type`: 解析器类型 (`auto`, `general`, `presentation`, `table`)
- `chunk_token_num`: 每个分块的最大 token 数 (1-2048)
- `delimiter`: 文本分割符 (默认: `\n。；！？`)
- `language`: 文档语言 (默认: `Chinese`)
- `layout_recognize`: 布局识别方法 (默认: `DeepDOC`)
- `zoomin`: OCR 缩放因子 (1-10)
- `from_page`: 起始页码 (默认: 0)
- `to_page`: 结束页码 (默认: 100000)

### 向量化参数
- `model_factory`: 模型工厂名称 (`BAAI`, `OpenAI`, `VLLM`, `LocalAI` 等)
- `model_name`: 具体的模型名称
- `api_key`: API 密钥 (某些模型需要)
- `base_url`: 服务端点 URL (本地模型需要)
- `batch_size`: 批处理大小 (默认: 16)
- `filename_embd_weight`: 文件名嵌入权重 (默认: 0.1)

## 🌟 使用场景

### 场景1: 仅文档分块
```bash
# 只需要分块，不需要向量化
curl -X POST "http://localhost:8089/chunk" \
  -F "file=@document.pdf" \
  -F "chunk_token_num=512"
```

### 场景2: 仅向量化已有分块
```bash
# 对已有分块进行向量化
curl -X POST "http://localhost:8090/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [...],
    "model_factory": "VLLM",
    "model_name": "bge-m3",
    "base_url": "http://localhost:8002/v1"
  }'
```

### 场景3: 一站式处理
```bash
# 一次完成分块和向量化
curl -X POST "http://localhost:8090/process" \
  -F "file=@document.pdf" \
  -F "model_factory=VLLM" \
  -F "model_name=bge-m3" \
  -F "base_url=http://localhost:8002/v1" \
  -F "chunk_token_num=512"
```

## 🚨 错误响应示例

### 文件格式不支持
```json
{
  "detail": "不支持的文件格式。支持的格式: .pdf, .docx, .doc, .txt, .md, .html, .pptx, .xlsx"
}
```

### 文件大小超限
```json
{
  "detail": "文件大小超过限制 (100MB)"
}
```

### 处理失败
```json
{
  "detail": "文档分块处理失败: 无法解析PDF文件"
}
```

### 模型参数缺失
```json
{
  "detail": "必须指定 model_factory 和 model_name 参数"
}
```

## 📊 API 文档

访问交互式 API 文档：
- 分块服务: http://localhost:8089/docs
- 向量化服务: http://localhost:8090/docs
