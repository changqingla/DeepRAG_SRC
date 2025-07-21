# DeepRAG 召回模块

基于 DeepRAG 原有算法的纯净召回系统，提供高质量的文档检索和重排序功能。

## 🚀 特性

- **完整的混合搜索**：结合文本搜索和向量搜索，支持权重调节
- **智能重排序**：支持多种重排序模型，提升检索精度
- **灵活的向量化模型**：支持 VLLM、OpenAI、HuggingFace 等多种向量化服务
- **降级策略**：当主要搜索策略失败时自动降级，确保系统稳定性
- **高亮显示**：自动高亮匹配的关键词
- **多种输出格式**：支持文本、JSON、简单格式输出

## 📋 系统要求

- Python 3.8+
- Elasticsearch 7.x/8.x
- 向量化模型服务（可选）
- 重排序模型服务（可选）

## 🛠️ 安装

```bash
# 克隆项目
git clone <repository-url>
cd DeepRAG_SRC/recall

# 安装依赖
pip install -r ../requirements.txt
```

## ⚙️ 配置

### Elasticsearch 配置

默认连接到 `http://10.0.100.36:9201`，可通过环境变量或代码修改：

```python
es_config = {
    "hosts": "http://your-es-host:9200",
    "timeout": 600
}
```

### 向量化模型配置

支持多种向量化模型工厂：

- **VLLM**: 高性能推理服务
- **OpenAI-API-Compatible**: OpenAI 兼容接口
- **HuggingFace**: HuggingFace 模型
- **LocalAI**: 本地 AI 服务

### 重排序模型配置

支持的重排序模型：

- **OpenAI-API-Compatible**: OpenAI 兼容的重排序服务
- **Jina**: Jina AI 重排序服务
- **Cohere**: Cohere 重排序服务
- **NVIDIA**: NVIDIA 重排序服务

## 🎯 快速开始

### 基本使用

```bash
# 简单查询
python recall_cli.py "什么是人工智能？"

# 指定索引
python recall_cli.py "机器学习" --indices my_index

# 调整参数
python recall_cli.py "深度学习" --top-k 20 --similarity 0.3 --vector-weight 0.7
```

### 使用向量化模型

```bash
python recall_cli.py "自然语言处理" \
  --model-factory VLLM \
  --model-name bge-m3 \
  --model-base-url http://10.0.1.4:8002/v1
```

### 使用重排序模型

```bash
python recall_cli.py "计算机视觉" \
  --model-factory VLLM \
  --model-name bge-m3 \
  --model-base-url http://10.0.1.4:8002/v1 \
  --rerank-factory "OpenAI-API-Compatible" \
  --rerank-model-name bge-reranker-v2-m3 \
  --rerank-base-url http://10.0.1.4:8001/v1
```

### 输出到文件

```bash
# JSON 格式
python recall_cli.py "数据挖掘" --output result.json --format json

# 简单格式
python recall_cli.py "推荐系统" --output result.txt --format simple
```

## 📖 API 使用

### Python API

```python
from deeprag_pure_retriever import deepragPureRetriever, deepragRetrievalConfig
from rag.llm import EmbeddingModel

# 创建配置
config = deepragRetrievalConfig(
    index_names=["my_index"],
    page_size=10,
    similarity_threshold=0.2,
    vector_similarity_weight=0.7
)

# 创建召回器
retriever = deepragPureRetriever(config)

# 创建向量化模型
embedding_model = EmbeddingModel["VLLM"](
    "empty",
    "bge-m3",
    "http://10.0.1.4:8002/v1"
)

# 执行召回
result = retriever.retrieval(
    question="什么是机器学习？",
    embd_mdl=embedding_model,
    page=1,
    page_size=10
)

# 获取结果
chunks = result["chunks"]
total = result["total"]
```

## 🔧 命令行参数

### 基本参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `question` | - | - | 查询问题 |
| `--indices` | `-i` | `deeprag_vectors` | ES索引名称列表 |
| `--top-k` | `-k` | `10` | 返回top-k结果 |
| `--similarity` | `-s` | `0.2` | 相似度阈值 |
| `--vector-weight` | `-w` | `0.3` | 向量相似度权重 |
| `--page` | - | `1` | 页码 |

### 向量化模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-factory` | `VLLM` | 向量化模型工厂 |
| `--model-name` | `bge-m3` | 向量化模型名称 |
| `--model-base-url` | `http://10.0.1.4:8002/v1` | 向量化模型服务地址 |

### 重排序模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rerank-factory` | - | 重排序模型工厂 |
| `--rerank-model-name` | - | 重排序模型名称 |
| `--rerank-base-url` | - | 重排序模型服务地址 |
| `--rerank-api-key` | - | 重排序模型API密钥 |

### 输出参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | - | 输出文件路径 |
| `--format` | - | `text` | 输出格式 (text/json/simple) |
| `--no-highlight` | - | - | 禁用高亮显示 |
| `--verbose` | `-v` | - | 详细输出 |

### 其他参数

| 参数 | 说明 |
|------|------|
| `--health` | 检查健康状态 |
| `--dry-run` | 干运行模式（不连接ES） |

## 🏗️ 架构设计

### 核心组件

1. **deepragPureRetriever**: 主要的召回器类
   - 实现完整的混合搜索算法
   - 支持文本搜索 + 向量搜索融合
   - 包含智能重排序逻辑

2. **ESAdapter**: Elasticsearch 适配器
   - 将独立的 ES 连接适配为 DeepRAG 接口
   - 实现 DocStoreConnection 抽象接口

3. **deepragRetrievalConfig**: 配置类
   - 封装所有召回相关的配置参数
   - 支持灵活的参数调整

### 搜索流程

```
查询输入 → 文本分析 → 向量编码
    ↓
混合搜索 (文本搜索 + 向量搜索)
    ↓
结果融合 (FusionExpr)
    ↓
重排序 (可选)
    ↓
分页和格式化
    ↓
输出结果
```

### 重排序策略

1. **页面限制**: 前3页使用重排序，后续页面直接使用ES排序
2. **模型重排序**: 使用专门的重排序模型计算语义相似度
3. **默认重排序**: 基于混合相似度的重排序算法
4. **降级策略**: 重排序失败时自动降级到默认算法

## 🔍 高级功能

### 降级策略

当主要搜索策略失败时，系统会自动：

1. 降低文本匹配阈值 (min_match: 0.3 → 0.1)
2. 提高向量相似度阈值 (similarity: 0.1 → 0.17)
3. 移除文档ID限制
4. 重新执行搜索

### 相似度计算

支持多种相似度计算方式：

- **文本相似度**: 基于关键词匹配和TF-IDF
- **向量相似度**: 基于余弦相似度
- **混合相似度**: 文本和向量的加权组合
- **重排序相似度**: 基于深度学习模型的语义相似度

### 高亮功能

自动识别和高亮匹配的关键词：

- 支持中文分词
- 细粒度关键词匹配
- 上下文保留

## 🐛 故障排除

### 常见问题

1. **连接 Elasticsearch 失败**
   ```
   解决方案: 检查 ES 服务状态和网络连接
   ```

2. **向量化模型连接失败**
   ```
   解决方案: 检查模型服务地址和API密钥
   ```

3. **重排序模型失败**
   ```
   解决方案: 系统会自动降级到默认重排序算法
   ```

4. **IK 分词器警告**
   ```
   解决方案: 系统会自动使用标准分词器，不影响功能
   ```

### 日志调试

```bash
# 启用详细日志
python recall_cli.py "查询内容" --verbose

# 检查健康状态
python recall_cli.py --health

# 干运行测试
python recall_cli.py "测试查询" --dry-run
```

## 📊 性能优化

### 建议配置

- **小数据集** (< 10万文档): `vector_weight=0.3`, `similarity=0.2`
- **大数据集** (> 100万文档): `vector_weight=0.7`, `similarity=0.1`
- **精确搜索**: 启用重排序模型
- **快速搜索**: 禁用重排序，使用默认算法

### 性能指标

- **搜索延迟**: 通常 < 500ms
- **重排序延迟**: 额外 100-300ms
- **吞吐量**: 支持并发查询

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache 2.0 License

## 👥 作者

- Hu Tao

---

更多信息请参考项目文档或联系开发团队。