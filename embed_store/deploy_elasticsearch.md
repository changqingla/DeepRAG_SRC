# 本地部署Elasticsearch指南

## 方法一：Docker部署（推荐）

### 1. 创建docker-compose.yml
```yaml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.11.3
    container_name: ragflow-es
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    networks:
      - ragflow-net

volumes:
  es_data:
    driver: local

networks:
  ragflow-net:
    driver: bridge
```

### 2. 启动ES
```bash
# 启动
docker-compose up -d

# 检查状态
curl http://localhost:9200
```

## 方法二：直接Docker运行
```bash
docker run -d \
  --name ragflow-es \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  elasticsearch:8.11.3
```

## 方法三：本地安装

### Ubuntu/Debian
```bash
# 安装Java
sudo apt update
sudo apt install openjdk-11-jdk

# 下载ES
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.3-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.11.3-linux-x86_64.tar.gz
cd elasticsearch-8.11.3

# 配置
echo "xpack.security.enabled: false" >> config/elasticsearch.yml
echo "discovery.type: single-node" >> config/elasticsearch.yml

# 启动
./bin/elasticsearch
```

### macOS
```bash
# 使用Homebrew
brew install elasticsearch

# 配置
echo "xpack.security.enabled: false" >> /usr/local/etc/elasticsearch/elasticsearch.yml
echo "discovery.type: single-node" >> /usr/local/etc/elasticsearch/elasticsearch.yml

# 启动
brew services start elasticsearch
```

## 验证安装
```bash
# 检查ES状态
curl http://localhost:9200

# 应该返回类似：
{
  "name" : "node-1",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "...",
  "version" : {
    "number" : "8.11.3"
  }
}
```

## ES配置说明
- **端口**: 9200 (HTTP), 9300 (传输)
- **安全**: 已禁用（适合本地开发）
- **模式**: 单节点模式
- **内存**: 1GB堆内存（可根据需要调整）
