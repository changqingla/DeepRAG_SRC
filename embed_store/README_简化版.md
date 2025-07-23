# 简化版文档存储模块

## 📖 概述

这是一个完全重写的简化版文档存储模块，专注于将解析后的文档分块存储到Elasticsearch中。去除了复杂的处理逻辑，保留核心功能。

## 🏗️ 架构

### 核心文件
- `es_connection.py` - 简化的ES连接类
- `chunk_store.py` - 文档存储器
- `test_store.py` - 测试脚本

### 设计原则
- **简单优先**: 去除复杂的租户、知识库层级
- **专注核心**: 专注于ES连接和文档存储
- **易于使用**: 提供简单的API和命令行接口

## 🚀 快速开始

### 1. 启动Elasticsearch

```bash
# Docker方式启动ES
docker run -d \
  --name es-simple \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.3

# 验证ES运行
curl http://localhost:9200
```

### 2. 存储文档

#### 命令行方式
```bash
# 直接运行存储
python chunk_store.py markdown_chunks_embedded.json

# 或者运行测试
python test_store.py markdown_chunks_embedded.json
```

#### 编程方式
```python
from chunk_store import DocumentStore

# 创建存储器
store = DocumentStore(
    es_host="http://localhost:9200",
    index_name="my_documents"
)

# 从文件存储
success_count, errors = store.load_and_store_from_file(
    "markdown_chunks_embedded.json",
    batch_size=100
)

print(f"成功存储 {success_count} 个分块")
```

## 📊 数据格式

### 输入格式
支持标准的分块JSON格式：
```json
[
  {
    "docnm_kwd": "document.pdf",
    "title_tks": "标题",
    "content_with_weight": "文档内容...",
    "content_ltks": "分词结果...",
    "q_1024_vec": [0.1, 0.2, ...],  // 向量数据
    "page_num_int": [1],
    "position_int": [[1, 0, 0, 0, 0]]
  }
]
```

### 存储格式
自动标准化为ES友好格式：
```json
{
  "id": "uuid",
  "doc_name": "document.pdf",
  "title": "标题",
  "content": "文档内容...",
  "content_tokens": "分词结果...",
  "vector_1024": [0.1, 0.2, ...],
  "page_num": 1,
  "position": 0,
  "create_time": "2024-01-01T12:00:00",
  "chunk_index": 0
}
```

## 🔧 API 参考

### DocumentStore 类

#### 初始化
```python
store = DocumentStore(
    es_host="http://localhost:9200",  # ES地址
    index_name="documents",           # 索引名称
    username="user",                  # 可选：用户名
    password="pass",                  # 可选：密码
    timeout=60                        # 可选：超时时间
)
```

#### 主要方法

**存储文档**
```python
# 从文件存储
success_count, errors = store.load_and_store_from_file(
    file_path="data.json",
    batch_size=100,
    progress_callback=callback_func
)

# 直接存储分块列表
success_count, errors = store.store_chunks(
    chunks=chunk_list,
    batch_size=100
)
```

**搜索文档**
```python
results = store.search_documents(
    query_text="搜索关键词",
    size=10,
    doc_name="特定文档.pdf"  # 可选过滤
)
```

**获取统计**
```python
stats = store.get_document_stats()
print(f"总分块数: {stats['total_chunks']}")
print(f"文档数量: {stats['unique_documents']}")
```

**索引管理**
```python
# 检查索引是否存在
exists = store.index_exists()

# 删除索引
success = store.delete_index()

# 获取ES健康状态
health = store.get_health()
```

## 🔍 搜索功能

### 文本搜索
```python
# 基本搜索
results = store.search_documents("人工智能")

# 限制文档范围
results = store.search_documents(
    "机器学习", 
    doc_name="AI教程.pdf"
)

# 搜索结果包含评分
for result in results:
    print(f"文档: {result['doc_name']}")
    print(f"评分: {result['_score']}")
    print(f"内容: {result['content'][:100]}...")
```

## 📈 性能优化

### 批量大小调整
```python
# 高性能机器
store.store_chunks(chunks, batch_size=200)

# 低配置机器
store.store_chunks(chunks, batch_size=50)
```

### 进度监控
```python
def my_progress(progress, message):
    print(f"{progress:.1%}: {message}")

store.load_and_store_from_file(
    "large_file.json",
    progress_callback=my_progress
)
```

## 🛠️ 故障排除

### 常见问题

**1. ES连接失败**
```
ConnectionError: [Errno 111] Connection refused
```
解决：检查ES是否运行，端口是否正确

**2. 索引创建失败**
```
RequestError: [400] resource_already_exists_exception
```
解决：索引已存在，使用 `store.delete_index()` 删除后重试

**3. 向量维度错误**
```
ValueError: 未找到向量字段或向量为空
```
解决：检查JSON文件是否包含 `q_*_vec` 格式的向量字段

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 现在会显示详细的调试信息
store = DocumentStore(...)
```

## 🎯 与原版的区别

| 特性 | 原版 | 简化版 |
|------|------|--------|
| 租户系统 | ✅ | ❌ 移除 |
| 知识库层级 | ✅ | ❌ 移除 |
| 复杂配置 | ✅ | ❌ 简化 |
| 核心存储 | ✅ | ✅ 保留 |
| 向量索引 | ✅ | ✅ 保留 |
| 搜索功能 | ✅ | ✅ 简化版 |
| 易用性 | ⚠️ 复杂 | ✅ 简单 |

## 📝 示例

完整的使用示例请参考 `test_store.py` 文件。
