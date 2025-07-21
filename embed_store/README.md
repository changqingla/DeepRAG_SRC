# RAGFlow Embedding Storage Module

基于RAGFlow原有存储逻辑的向量化分块存储模块，将已向量化的文档分块存储到Elasticsearch中，不使用租户系统，直接利用RAGFlow的存储算法。

## 功能特点

- **完全复用RAGFlow存储逻辑**: 使用RAGFlow原有的Elasticsearch存储算法
- **无租户系统依赖**: 独立运行，不需要RAGFlow的租户配置
- **批量存储**: 高效的批量插入操作
- **索引自动创建**: 自动创建和管理Elasticsearch索引
- **数据验证**: 内置分块数据验证和修复功能
- **存储分析**: 详细的存储结果分析和报告
- **灵活配置**: 支持多种存储配置方式

## 存储架构

### Elasticsearch索引结构

基于RAGFlow的mapping配置(`conf/mapping.json`)：

```json
{
  "mappings": {
    "dynamic_templates": [
      {
        "dense_vector": {
          "match": "*_1024_vec",
          "mapping": {
            "type": "dense_vector",
            "index": true,
            "similarity": "cosine",
            "dims": 1024
          }
        }
      }
    ]
  }
}
```

### 存储的字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | keyword | 分块唯一ID (xxhash生成) |
| `doc_id` | keyword | 文档ID |
| `kb_id` | keyword | 知识库ID |
| `content_with_weight` | text | 分块内容 |
| `content_ltks` | text | 基础分词结果 |
| `content_sm_ltks` | text | 细粒度分词结果 |
| `docnm_kwd` | keyword | 文档名称 |
| `title_tks` | text | 标题分词 |
| `page_num_int` | integer[] | 页码信息 |
| `position_int` | integer[] | 位置坐标 |
| `q_*_vec` | dense_vector | 向量数据 |
| `create_time` | date | 创建时间 |
| `create_timestamp_flt` | float | 创建时间戳 |

## 快速开始

### 基本使用

```python
from embed_store.chunk_store import ChunkStore, ChunkStoreConfig

# 创建存储配置
config = ChunkStoreConfig(
    index_name="my_documents",
    kb_id="kb123",
    batch_size=4
)

# 初始化存储器
store = ChunkStore(config)

# 存储分块 (chunks来自embedding模块的输出)
stored_count, errors = store.store_chunks(chunks)

print(f"成功存储 {stored_count} 个分块")
```

### 命令行使用

```bash
# 基本存储
python embed_store/store_cli.py embedded_chunks.json

# 使用自定义索引名称
python embed_store/store_cli.py embedded_chunks.json --index-name my_docs

# 使用租户ID (RAGFlow风格)
python embed_store/store_cli.py embedded_chunks.json --tenant-id user123

# 验证分块数据
python embed_store/store_cli.py embedded_chunks.json --validate-only

# 修复分块问题
python embed_store/store_cli.py embedded_chunks.json --fix-chunks

# 自定义批量大小
python embed_store/store_cli.py embedded_chunks.json --batch-size 8
```

## 详细使用说明

### 1. 存储配置

#### 默认配置
```python
from embed_store.store_utils import StorageConfigManager

# 创建默认配置
config = StorageConfigManager.create_default_config("my_index")

# 创建租户风格配置
config = StorageConfigManager.create_tenant_config("user123")
```

#### 自定义配置
```python
from embed_store.chunk_store import ChunkStoreConfig

config = ChunkStoreConfig(
    index_name="custom_index",
    kb_id="custom_kb",
    doc_id="custom_doc",
    batch_size=8,
    auto_create_index=True
)
```

### 2. 数据验证

#### 验证分块数据
```python
from embed_store.store_utils import ChunkValidator

# 验证分块
validation = ChunkValidator.validate_chunks(chunks)
print(f"验证结果: {validation['valid']}")
print(f"错误: {validation['errors']}")
print(f"警告: {validation['warnings']}")

# 修复分块问题
if not validation['valid']:
    fixed_chunks = ChunkValidator.fix_chunks(chunks)
```

#### 必需字段检查
- `content_with_weight`: 分块内容 (必需)
- `q_*_vec`: 向量数据 (必需，格式: `q_1024_vec`, `q_768_vec`等)

### 3. 存储操作

#### 批量存储
```python
store = ChunkStore(config)

# 存储分块
stored_count, error_messages = store.store_chunks(chunks)

if error_messages:
    print(f"存储错误: {error_messages}")
else:
    print(f"成功存储 {stored_count} 个分块")
```

#### 存储结果分析
```python
from embed_store.store_utils import StorageResult, StorageAnalyzer

# 创建结果对象
result = StorageResult(
    stored_count=stored_count,
    total_count=len(chunks),
    error_count=len(error_messages),
    error_messages=error_messages,
    processing_time=processing_time,
    index_info=store.get_index_info()
)

# 生成分析报告
report = StorageAnalyzer.create_storage_report(result)
print(report)
```

### 4. 索引管理

#### 索引信息
```python
# 获取索引信息
index_info = store.get_index_info()
print(f"索引名称: {index_info['index_name']}")
print(f"向量维度: {index_info['vector_size']}")
print(f"索引存在: {index_info['index_exists']}")
```

#### 索引操作
```python
# 检查索引是否存在
exists = store.es_conn.index_exists(index_name, kb_id)

# 删除索引 (谨慎使用!)
store.delete_index()
```

## 与embedding模块集成

完整的工作流程：

```python
# 1. 使用embedding模块生成向量
from embedding.chunk_embedder import ChunkEmbedder
embedder = ChunkEmbedder()
token_count, vector_size = embedder.embed_chunks_sync(chunks)

# 2. 使用embed_store模块存储到ES
from embed_store.chunk_store import ChunkStore, ChunkStoreConfig
config = ChunkStoreConfig(index_name="my_docs")
store = ChunkStore(config)
stored_count, errors = store.store_chunks(chunks)

print(f"向量化: {token_count} tokens, {vector_size}维")
print(f"存储: {stored_count} 个分块")
```

## 命令行工具详解

### 基本命令

```bash
# 存储分块
python embed_store/store_cli.py embedded_chunks.json

# 显示帮助
python embed_store/store_cli.py --help
```

### 配置选项

```bash
# 索引配置
--index-name my_docs          # 自定义索引名称
--tenant-id user123           # 租户ID (生成ragflow_user123索引)
--kb-id kb123                 # 知识库ID
--doc-id doc456               # 文档ID
--batch-size 8                # 批量大小

# 索引管理
--no-auto-create              # 不自动创建索引
--show-index-info             # 显示索引信息
--delete-index                # 删除索引 (危险操作!)
```

### 验证选项

```bash
# 数据验证
--validate-only               # 仅验证，不存储
--fix-chunks                  # 尝试修复分块问题
```

### 输出选项

```bash
# 报告和导出
--output-report report.txt    # 保存存储报告
--export-mapping mapping.json # 导出索引映射信息
--export-sample sample.json   # 导出样本分块结构
```

### 配置管理

```bash
# 配置文件
--save-config config.json     # 保存配置
--load-config config.json     # 加载配置
```

## 错误处理

### 常见错误

1. **Elasticsearch连接失败**
   ```
   解决方案: 检查ES服务状态和配置
   ```

2. **索引创建失败**
   ```
   解决方案: 检查ES权限和mapping配置
   ```

3. **分块验证失败**
   ```
   解决方案: 使用--fix-chunks自动修复或手动检查数据
   ```

4. **批量插入错误**
   ```
   解决方案: 减少batch_size或检查数据格式
   ```

### 错误日志

启用详细日志：
```bash
python embed_store/store_cli.py embedded_chunks.json --verbose
```

## 性能优化

### 批量大小调整

```python
# 小内存环境
config = ChunkStoreConfig(batch_size=2)

# 大内存环境
config = ChunkStoreConfig(batch_size=16)
```

### 索引优化

- 使用SSD存储提高ES性能
- 调整ES的`refresh_interval`设置
- 合理设置ES的分片数量

## 监控和分析

### 存储性能监控

```python
# 性能分析
analysis = StorageAnalyzer.analyze_storage_result(result)
print(f"存储速度: {analysis['performance']['chunks_per_second']:.1f} chunks/s")
print(f"成功率: {analysis['success_rate']:.1%}")
```

### 存储报告

```bash
# 生成详细报告
python embed_store/store_cli.py embedded_chunks.json --output-report storage_report.txt
```

## 示例代码

查看 `example.py` 文件获取完整的使用示例，包括：
- 基本存储操作
- 租户风格配置
- 数据验证和修复
- 存储结果分析
- 配置管理
- 完整集成工作流程

## 注意事项

1. **数据备份**: 存储前备份重要数据
2. **索引命名**: 使用有意义的索引名称
3. **权限管理**: 确保ES访问权限正确
4. **资源监控**: 监控ES集群资源使用情况
5. **版本兼容**: 确保ES版本 >= 8.0

## 许可证

Apache 2.0 License - 与RAGFlow保持一致
