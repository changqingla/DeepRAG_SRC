# DeepRAG 文档嵌入模块

## 📖 概述

DeepRAG 文档嵌入模块提供了文档分块嵌入功能，该模块为文档分块生成高质量的向量表示。

## ✨ 主要特性

- **🔧 多模型支持**: 支持 三十 多种嵌入模型（VLLM、OpenAI、LocalAI、通义千问、智谱AI等）
- **⚡ 批量处理**: 高效的批量嵌入处理，支持自定义批次大小
- **🎯 明确配置**: 必须明确指定模型参数，避免意外使用错误模型
- **📊 结果分析**: 内置嵌入结果分析和相似度计算
- **🔍 聚类功能**: 基于嵌入向量的文档聚类
- **💾 多格式导出**: 支持多种格式的结果导出
- **🖥️ 命令行工具**: 完整的命令行界面，方便批量操作

## 📁 模块结构

```
embedding/
├── README.md                    # 本文档
├── __init__.py                  # 模块初始化文件
├── chunk_embedder.py           # 核心嵌入器类
├── embed_cli.py                # 命令行工具
├── embedding_utils.py          # 工具函数和辅助类
├── markdown_chunks.json        # 示例分块数据
└── markdown_chunks_embedded.json # 示例嵌入结果
```

## 🚀 快速开始


### 1. 基本使用

```python
from chunk_embedder import ChunkEmbedder, EmbeddingConfig

# 必须明确指定模型配置
config = EmbeddingConfig(
    model_factory="VLLM",
    model_name="bge-m3",
    base_url="http://localhost:8002/v1"
)
embedder = ChunkEmbedder(config)

# 加载文档分块
chunks = [
    {
        "content_with_weight": "这是第一个文档分块的内容",
        "docnm_kwd": "document1.pdf"
    },
    {
        "content_with_weight": "这是第二个文档分块的内容",
        "docnm_kwd": "document1.pdf"
    }
]

# 生成嵌入向量
token_count, vector_size = embedder.embed_chunks_sync(chunks)
print(f"处理了 {token_count} 个token，向量维度: {vector_size}")

# 查看嵌入结果
for chunk in chunks:
    vector_field = embedder.get_embedding_field_name()
    print(f"分块嵌入向量维度: {len(chunk[vector_field])}")
```

### 2. 使用不同的嵌入模型

```python
# OpenAI 模型
openai_config = EmbeddingConfig(
    model_factory="OpenAI",
    model_name="text-embedding-3-small",
    api_key="your-openai-api-key"
)
embedder = ChunkEmbedder(openai_config)

# 通义千问模型
qwen_config = EmbeddingConfig(
    model_factory="Tongyi-Qianwen",
    model_name="text_embedding_v2",
    api_key="your-qwen-api-key"
)
embedder = ChunkEmbedder(qwen_config)

# 智谱AI模型
zhipu_config = EmbeddingConfig(
    model_factory="ZHIPU-AI",
    model_name="embedding-2",
    api_key="your-zhipu-api-key"
)
embedder = ChunkEmbedder(zhipu_config)

# LocalAI 模型
localai_config = EmbeddingConfig(
    model_factory="LocalAI",
    model_name="bge-m3",
    base_url="http://localhost:8080/v1"
)
embedder = ChunkEmbedder(localai_config)

```

### 3. 命令行使用

```bash
# 使用 VLLM 模型
python embed_cli.py chunks.json --model-factory VLLM --model-name bge-m3 --model-base-url http://10.0.1.4:8002/v1

# 使用 LocalAI 模型
python embed_cli.py chunks.json --model-factory LocalAI --model-name bge-m3 --model-base-url http://localhost:8080/v1

# 使用 OpenAI 模型
python embed_cli.py chunks.json --model-factory OpenAI --model-name text-embedding-3-small --api-key YOUR_KEY

# 使用通义千问模型
python embed_cli.py chunks.json --model-factory Tongyi-Qianwen --model-name text_embedding_v2 --api-key YOUR_KEY

# 使用智谱AI模型
python embed_cli.py chunks.json --model-factory ZHIPU-AI --model-name embedding-2 --api-key YOUR_KEY

# 分析嵌入结果
python embed_cli.py chunks.json --analyze --output analysis.json --model-factory VLLM --model-name bge-m3 --model-base-url http://10.0.1.4:8002/v1

# 查找相似分块
python embed_cli.py chunks.json --find-similar 5 --model-factory VLLM --model-name bge-m3 --model-base-url http://10.0.1.4:8002/v1

# 聚类分析
python embed_cli.py chunks.json --cluster 3 --model-factory VLLM --model-name bge-m3 --model-base-url http://10.0.1.4:8002/v1

# 导出向量
python embed_cli.py chunks.json --export-vectors vectors.npy --model-factory VLLM --model-name bge-m3 --model-base-url http://10.0.1.4:8002/v1
```

## 🔧 配置选项

### 必需的配置参数

本模块要求明确指定所有模型参数，不再支持默认配置。

```python
# 必需参数
config = EmbeddingConfig(
    model_factory="VLLM",           # 必需: 模型工厂名称
    model_name="bge-m3",            # 必需: 模型名称

    # 根据模型类型必需的参数
    api_key="your-api-key",         # 云服务 API 通常需要
    base_url="http://localhost:8002/v1",  # 本地服务需要

    # 可选参数
    filename_embd_weight=0.1,       # 文件名嵌入权重
    batch_size=16                   # 批处理大小
)
```


## 📊 分析功能

### 嵌入质量分析

```python
from embedding_utils import EmbeddingAnalyzer

# 分析嵌入结果
analysis = EmbeddingAnalyzer.analyze_embeddings(embedded_chunks)
print(f"向量维度: {analysis['vector_dimension']}")
print(f"平均相似度: {analysis['similarity_stats']['mean']:.4f}")
```

### 相似度搜索

```python
# 查找与第一个分块最相似的5个分块
similar_chunks = EmbeddingAnalyzer.find_similar_chunks(
    embedded_chunks[1:],  # 搜索范围
    embedded_chunks[0],   # 查询分块
    top_k=5
)

for idx, similarity, chunk in similar_chunks:
    print(f"相似度: {similarity:.4f} - {chunk['content_with_weight'][:50]}...")
```

### 聚类分析

```python
# 将分块聚类为3个类别
cluster_result = EmbeddingAnalyzer.cluster_chunks(embedded_chunks, n_clusters=3)
print(f"轮廓系数: {cluster_result['silhouette_score']:.4f}")
print(f"聚类大小: {cluster_result['cluster_sizes']}")
```

## 💾 导出功能

### 导出向量

```python
from embedding_utils import EmbeddingExporter
from pathlib import Path

# 导出为 NumPy 数组
EmbeddingExporter.export_vectors_only(
    embedded_chunks,
    Path("vectors.npy"),
    format="npy"
)

# 导出为 CSV
EmbeddingExporter.export_vectors_only(
    embedded_chunks,
    Path("vectors.csv"),
    format="csv"
)
```

### 导出元数据

```python
# 导出带元数据的分块信息
EmbeddingExporter.export_with_metadata(
    embedded_chunks,
    Path("chunks_with_metadata.json"),
    include_vectors=True
)
```

## 🛠️ 命令行工具详解

### 基本命令

```bash
# 查看帮助
python embed_cli.py --help

# 正确的使用方式 - 必须指定模型
python embed_cli.py chunks.json --model-factory VLLM --model-name bge-m3 --model-base-url http://localhost:8002/v1

# 指定输出文件
python embed_cli.py chunks.json --model-factory VLLM --model-name bge-m3 --model-base-url http://localhost:8002/v1 --output result.json
```

### 高级功能

```bash
# 详细日志输出
python embed_cli.py chunks.json --verbose

# 自定义批处理大小
python embed_cli.py chunks.json --batch-size 32

# 调整文件名权重
python embed_cli.py chunks.json --filename-weight 0.2

# 组合多个功能
python embed_cli.py chunks.json \
    --model-factory VLLM \
    --model-name bge-m3 \
    --model-base-url http://localhost:8002/v1 \
    --analyze \
    --find-similar 10 \
    --cluster 5 \
    --export-vectors vectors.npy \
    --verbose
```

## 📋 输入数据格式

输入的 JSON 文件应包含分块数组，每个分块至少需要以下字段：

```json
[
  {
    "content_with_weight": "分块的文本内容",
    "docnm_kwd": "文档名称",
    "title_tks": "标题关键词（可选）",
    "question_kwd": ["问题关键词数组（可选）"]
  }
]
```

## 📤 输出数据格式

嵌入后的分块会添加向量字段：

```json
[
  {
    "content_with_weight": "分块的文本内容",
    "docnm_kwd": "文档名称",
    "q_1024_vec": [0.1, -0.2, 0.3, ...],  // 1024维向量
    // ... 其他原有字段
  }
]
```
## 📄 许可证

Apache 2.0 License

---

**作者**: HU TAO
**更新时间**: 2025-01-17