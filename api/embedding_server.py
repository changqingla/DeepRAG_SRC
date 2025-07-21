"""
DeepRAG 异步向量化服务

基于 embedding 模块实现的高并发 HTTP 向量化接口，支持：
- 异步批量文档分块向量化
- 多种嵌入模型支持
- 并发控制和资源管理
- RESTful API 接口
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入 embedding 模块
from embedding.chunk_embedder import ChunkEmbedder, EmbeddingConfig
from embedding.embedding_utils import EmbeddingAnalyzer

# chunk 模块将在需要时动态导入

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """向量化请求模型"""
    model_config = {"protected_namespaces": ()}

    chunks: List[Dict[str, Any]] = Field(..., description="文档分块列表")
    model_factory: str = Field(..., description="模型工厂名称")
    model_name: str = Field(..., description="模型名称")
    api_key: Optional[str] = Field(None, description="API 密钥")
    base_url: Optional[str] = Field(None, description="服务端点 URL")
    batch_size: int = Field(16, description="批处理大小")
    filename_embd_weight: float = Field(0.1, description="文件名嵌入权重")


class EmbeddingResponse(BaseModel):
    """向量化响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="包含向量的分块列表")
    stats: Optional[Dict[str, Any]] = Field(None, description="统计信息")
    processing_time: float = Field(..., description="处理时间（秒）")


class DocumentProcessResponse(BaseModel):
    """文档处理响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="包含向量的分块列表")
    chunk_stats: Optional[Dict[str, Any]] = Field(None, description="分块统计信息")
    embedding_stats: Optional[Dict[str, Any]] = Field(None, description="向量化统计信息")
    total_processing_time: float = Field(..., description="总处理时间（秒）")
    chunk_time: float = Field(..., description="分块处理时间（秒）")
    embedding_time: float = Field(..., description="向量化处理时间（秒）")


class EmbeddingService:
    """向量化服务类"""
    
    def __init__(self, max_concurrent_tasks: int = 100):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks = 0
        
    async def embed_chunks(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        异步向量化文档分块
        
        Args:
            request: 向量化请求
            
        Returns:
            EmbeddingResponse: 向量化结果
        """
        start_time = time.time()
        
        async with self.semaphore:
            self.active_tasks += 1
            try:
                logger.info(f"开始处理向量化请求，分块数量: {len(request.chunks)}")
                
                # 创建嵌入配置
                config = EmbeddingConfig(
                    model_factory=request.model_factory,
                    model_name=request.model_name,
                    api_key=request.api_key or "",
                    base_url=request.base_url,
                    batch_size=request.batch_size,
                    filename_embd_weight=request.filename_embd_weight
                )
                
                # 在线程池中运行同步的嵌入操作
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._sync_embed_chunks, 
                    config, 
                    request.chunks
                )
                
                processing_time = time.time() - start_time
                
                return EmbeddingResponse(
                    success=True,
                    message=f"成功向量化 {len(request.chunks)} 个分块",
                    chunks=result["chunks"],
                    stats=result["stats"],
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"向量化失败: {e}")
                processing_time = time.time() - start_time
                
                return EmbeddingResponse(
                    success=False,
                    message=f"向量化失败: {str(e)}",
                    processing_time=processing_time
                )
            finally:
                self.active_tasks -= 1

    async def process_document(self,
                              file_content: bytes,
                              filename: str,
                              model_factory: str,
                              model_name: str,
                              api_key: Optional[str] = None,
                              base_url: Optional[str] = None,
                              parser_type: str = "general",
                              chunk_token_num: int = 256,
                              delimiter: str = "\n。；！？",
                              language: str = "Chinese",
                              layout_recognize: str = "DeepDOC",
                              from_page: int = 0,
                              to_page: int = 100000,
                              zoomin: int = 3,
                              batch_size: int = 16,
                              filename_embd_weight: float = 0.1) -> DocumentProcessResponse:
        """
        异步处理文档：分块 + 向量化

        Args:
            file_content: 文件内容
            filename: 文件名
            model_factory: 模型工厂名称
            model_name: 模型名称
            api_key: API 密钥
            base_url: 服务端点 URL
            parser_type: 解析器类型
            chunk_token_num: 每个分块的最大 token 数
            delimiter: 分块分隔符
            language: 文档语言
            layout_recognize: 布局识别方法
            from_page: 起始页码
            to_page: 结束页码
            zoomin: OCR 缩放因子
            batch_size: 批处理大小
            filename_embd_weight: 文件名嵌入权重

        Returns:
            DocumentProcessResponse: 处理结果
        """
        start_time = time.time()

        async with self.semaphore:
            self.active_tasks += 1
            try:
                logger.info(f"开始处理文档: {filename}")

                # 在线程池中运行同步的文档处理操作
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sync_process_document,
                    file_content,
                    filename,
                    model_factory,
                    model_name,
                    api_key,
                    base_url,
                    parser_type,
                    chunk_token_num,
                    delimiter,
                    language,
                    layout_recognize,
                    from_page,
                    to_page,
                    zoomin,
                    batch_size,
                    filename_embd_weight
                )

                total_processing_time = time.time() - start_time

                return DocumentProcessResponse(
                    success=True,
                    message=f"成功处理文档 {filename}，生成 {result['chunk_count']} 个向量化分块",
                    chunks=result["chunks"],
                    chunk_stats=result["chunk_stats"],
                    embedding_stats=result["embedding_stats"],
                    total_processing_time=total_processing_time,
                    chunk_time=result["chunk_time"],
                    embedding_time=result["embedding_time"]
                )

            except Exception as e:
                logger.error(f"文档处理失败: {e}")
                total_processing_time = time.time() - start_time

                return DocumentProcessResponse(
                    success=False,
                    message=f"文档处理失败: {str(e)}",
                    total_processing_time=total_processing_time,
                    chunk_time=0.0,
                    embedding_time=0.0
                )
            finally:
                self.active_tasks -= 1
    
    def _sync_embed_chunks(self, config: EmbeddingConfig, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        同步执行向量化操作

        Args:
            config: 嵌入配置
            chunks: 文档分块列表

        Returns:
            Dict: 包含向量化结果和统计信息的字典
        """
        # 创建嵌入器
        embedder = ChunkEmbedder(config)

        # 执行向量化
        token_count, vector_size = embedder.embed_chunks_sync(chunks)

        # 生成统计信息
        stats = {
            "total_chunks": len(chunks),
            "total_tokens": token_count,
            "vector_dimension": vector_size,
            "model_factory": config.model_factory,
            "model_name": config.model_name
        }

        return {
            "chunks": chunks,
            "stats": stats
        }

    def _sync_process_document(self,
                              file_content: bytes,
                              filename: str,
                              model_factory: str,
                              model_name: str,
                              api_key: Optional[str],
                              base_url: Optional[str],
                              parser_type: str,
                              chunk_token_num: int,
                              delimiter: str,
                              language: str,
                              layout_recognize: str,
                              from_page: int,
                              to_page: int,
                              zoomin: int,
                              batch_size: int,
                              filename_embd_weight: float) -> Dict[str, Any]:
        """
        同步执行文档处理操作：分块 + 向量化

        Args:
            file_content: 文件内容
            filename: 文件名
            其他参数: 分块和向量化配置参数

        Returns:
            Dict: 包含处理结果和统计信息的字典
        """
        import tempfile
        import os

        # 1. 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # 2. 动态导入 chunk 模块
            try:
                from chunk.document_chunker import DocumentChunker
            except ImportError as e:
                raise Exception(f"无法导入分块模块: {e}。请确保安装了所有依赖。")

            # 3. 执行分块
            chunk_start_time = time.time()
            chunker = DocumentChunker(
                parser_type=parser_type,
                chunk_token_num=chunk_token_num,
                delimiter=delimiter,
                language=language,
                layout_recognize=layout_recognize,
                from_page=from_page,
                to_page=to_page,
                zoomin=zoomin
            )
            chunks = chunker.chunk_document(file_path=temp_file_path)
            chunk_time = time.time() - chunk_start_time

            logger.info(f"分块完成: {len(chunks)} 个分块，耗时 {chunk_time:.2f}s")

            # 4. 生成分块统计信息
            chunk_stats = {
                "total_chunks": len(chunks),
                "chunk_time": chunk_time,
                "parser_type": parser_type,
                "chunk_token_num": chunk_token_num
            }

            # 5. 创建嵌入配置
            embedding_config = EmbeddingConfig(
                model_factory=model_factory,
                model_name=model_name,
                api_key=api_key or "",
                base_url=base_url,
                batch_size=batch_size,
                filename_embd_weight=filename_embd_weight
            )

            # 6. 执行向量化
            embedding_start_time = time.time()
            embedder = ChunkEmbedder(embedding_config)
            token_count, vector_size = embedder.embed_chunks_sync(chunks)
            embedding_time = time.time() - embedding_start_time

            logger.info(f"向量化完成: {len(chunks)} 个分块，耗时 {embedding_time:.2f}s")

            # 7. 生成向量化统计信息
            embedding_stats = {
                "total_chunks": len(chunks),
                "total_tokens": token_count,
                "vector_dimension": vector_size,
                "model_factory": model_factory,
                "model_name": model_name,
                "embedding_time": embedding_time
            }

            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chunk_stats": chunk_stats,
                "embedding_stats": embedding_stats,
                "chunk_time": chunk_time,
                "embedding_time": embedding_time
            }

        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


# 全局服务实例
embedding_service = EmbeddingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动向量化服务...")
    yield
    logger.info("关闭向量化服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="DeepRAG 向量化服务",
    description="基于 DeepRAG 的异步高并发文档向量化 API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "DeepRAG 向量化服务",
        "version": "1.0.0",
        "status": "running",
        "active_tasks": embedding_service.active_tasks,
        "max_concurrent_tasks": embedding_service.max_concurrent_tasks
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "active_tasks": embedding_service.active_tasks,
        "max_concurrent_tasks": embedding_service.max_concurrent_tasks
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_chunks(request: EmbeddingRequest):
    """
    向量化文档分块

    接收文档分块列表，返回包含向量的分块列表。
    """
    if not request.chunks:
        raise HTTPException(status_code=400, detail="分块列表不能为空")

    if len(request.chunks) > 1000:
        raise HTTPException(status_code=400, detail="单次请求分块数量不能超过 1000")

    # 验证必需的模型参数
    if not request.model_factory or not request.model_name:
        raise HTTPException(
            status_code=400,
            detail="必须指定 model_factory 和 model_name 参数"
        )

    return await embedding_service.embed_chunks(request)


@app.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="要处理的文档文件"),
    model_factory: str = Form(..., description="模型工厂名称"),
    model_name: str = Form(..., description="模型名称"),
    api_key: str = Form("", description="API 密钥"),
    base_url: str = Form("", description="服务端点 URL"),
    parser_type: str = Form("general", description="解析器类型"),
    chunk_token_num: int = Form(256, description="每个分块的最大 token 数"),
    delimiter: str = Form("\n。；！？", description="分块分隔符"),
    language: str = Form("Chinese", description="文档语言"),
    layout_recognize: str = Form("DeepDOC", description="布局识别方法"),
    from_page: int = Form(0, description="起始页码"),
    to_page: int = Form(100000, description="结束页码"),
    zoomin: int = Form(3, description="OCR 缩放因子"),
    batch_size: int = Form(16, description="批处理大小"),
    filename_embd_weight: float = Form(0.1, description="文件名嵌入权重")
):
    """
    处理文档：分块 + 向量化

    上传原始文档文件，一次完成分块和向量化两个操作。
    支持的文件格式：PDF、Word、Excel、PowerPoint、Markdown、TXT 等
    """
    # 验证必需的模型参数
    if not model_factory or not model_name:
        raise HTTPException(
            status_code=400,
            detail="必须指定 model_factory 和 model_name 参数"
        )

    # 验证文件大小（限制为 100MB）
    max_file_size = 100 * 1024 * 1024  # 100MB
    file_content = await file.read()
    if len(file_content) > max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制（最大 {max_file_size // (1024*1024)}MB）"
        )

    # 验证文件格式
    supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.pptx', '.xlsx'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_extension}。支持的格式: {', '.join(supported_extensions)}"
        )

    return await embedding_service.process_document(
        file_content=file_content,
        filename=file.filename,
        model_factory=model_factory,
        model_name=model_name,
        api_key=api_key if api_key else None,
        base_url=base_url if base_url else None,
        parser_type=parser_type,
        chunk_token_num=chunk_token_num,
        delimiter=delimiter,
        language=language,
        layout_recognize=layout_recognize,
        from_page=from_page,
        to_page=to_page,
        zoomin=zoomin,
        batch_size=batch_size,
        filename_embd_weight=filename_embd_weight
    )


@app.get("/models")
async def list_supported_models():
    """列出支持的模型"""
    try:
        # 导入 EmbeddingModel 字典
        from rag.llm import EmbeddingModel
        
        models = {}
        for factory_name in EmbeddingModel.keys():
            models[factory_name] = {
                "factory_name": factory_name,
                "description": f"{factory_name} 嵌入模型"
            }
        
        return {
            "supported_models": models,
            "total_count": len(models)
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取模型列表")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepRAG 向量化服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8090, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--log-level", default="info", help="日志级别")

    args = parser.parse_args()

    logger.info(f"启动 DeepRAG 向量化服务...")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "embedding_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )
