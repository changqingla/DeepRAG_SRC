#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高并发异步文档分块 HTTP 服务

这个模块提供了一个基于 FastAPI 的高性能异步 HTTP 接口，用于文档分块处理。
支持多种文档格式，具有以下特性：
1. 异步处理 - 使用 asyncio 和 FastAPI 实现高并发
2. 直接响应 - 快速处理并直接返回结果
3. 智能处理 - 根据文件大小智能选择处理策略
4. 错误处理 - 完善的错误处理和重试机制
5. 监控指标 - 提供性能监控和统计信息

Author: DeepRAG Team
License: Apache 2.0
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime, timedelta
import tempfile
import shutil

# FastAPI 相关导入
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 添加项目根目录到路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入分块器
from chunk.document_chunker import DocumentChunker
from rag.utils import ParserType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# 确保所有相关模块的日志级别都是 INFO
logging.getLogger('rag.nlp').setLevel(logging.INFO)
logging.getLogger('rag.app.naive').setLevel(logging.INFO)
logging.getLogger('chunk.document_chunker').setLevel(logging.INFO)

# 全局配置
class Config:
    """服务配置"""
    MAX_WORKERS = 10  # 最大工作线程数
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    LARGE_FILE_SIZE = 10 * 1024 * 1024  # 10MB - 大文件阈值
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.pptx', '.xlsx'}
    MAX_CONCURRENT_TASKS = 50  # 最大并发任务数
    MAX_BATCH_SIZE = 10  # 批量处理最大文件数
    TEMP_DIR = "/tmp/deeprag_chunks"

# Pydantic 模型定义
class ChunkRequest(BaseModel):
    """分块请求模型"""
    parser_type: str = Field(default="auto", description="解析器类型")
    chunk_token_num: int = Field(default=256, ge=1, le=2048, description="每个分块的最大token数")
    delimiter: str = Field(default="\n。；！？", description="文本分割符")
    language: str = Field(default="Chinese", description="文档语言")
    layout_recognize: str = Field(default="DeepDOC", description="布局识别方法")
    zoomin: int = Field(default=3, ge=1, le=10, description="OCR缩放因子")
    from_page: int = Field(default=0, ge=0, description="起始页码")
    to_page: int = Field(default=100000, ge=1, description="结束页码")

class ChunkResponse(BaseModel):
    """分块响应模型"""
    success: bool = Field(description="处理是否成功")
    chunks: Optional[List[Dict[str, Any]]] = Field(default=None, description="分块结果")
    total_chunks: int = Field(default=0, description="总分块数")
    processing_time: float = Field(description="处理时间（秒）")
    file_size: int = Field(description="文件大小（字节）")
    parser_type: str = Field(description="使用的解析器类型")
    error: Optional[str] = Field(default=None, description="错误信息")

class BatchChunkResponse(BaseModel):
    """批量分块响应模型"""
    success: bool = Field(description="批量处理是否成功")
    results: List[Dict[str, Any]] = Field(description="每个文件的处理结果")
    total_files: int = Field(description="总文件数")
    successful_files: int = Field(description="成功处理的文件数")
    failed_files: int = Field(description="失败的文件数")
    total_processing_time: float = Field(description="总处理时间（秒）")

class ServiceStats(BaseModel):
    """服务统计信息"""
    uptime: float = Field(description="服务运行时间（秒）")
    total_requests: int = Field(description="总请求数")
    successful_requests: int = Field(description="成功请求数")
    failed_requests: int = Field(description="失败请求数")
    average_processing_time: float = Field(description="平均处理时间（秒）")
    current_concurrent_tasks: int = Field(description="当前并发任务数")

# 创建 FastAPI 应用
app = FastAPI(
    title="DeepRAG 文档分块服务",
    description="高并发异步文档分块 HTTP 接口",
    version="2.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_TASKS)

# 统计信息
stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "processing_times": [],
    "current_concurrent_tasks": 0
}

class ChunkService:
    """简化的分块服务类"""
    
    def __init__(self):
        self.temp_dir = Path(Config.TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def validate_file(file: UploadFile) -> bool:
        """验证上传文件"""
        if not file.filename:
            return False
        
        file_ext = Path(file.filename).suffix.lower()
        return file_ext in Config.SUPPORTED_FORMATS
    
    @staticmethod
    def detect_parser_type(filename: str) -> str:
        """根据文件名检测解析器类型"""
        ext = Path(filename).suffix.lower()
        
        parser_map = {
            '.pdf': "general",
            '.docx': "general",
            '.doc': "general",
            '.txt': "general",
            '.md': "general",
            '.html': "general",
            '.pptx': ParserType.PRESENTATION,
            '.xlsx': ParserType.TABLE,
        }
        
        return parser_map.get(ext, "general")
    
    def save_temp_file(self, file_content: bytes, filename: str) -> str:
        """保存临时文件"""
        file_id = str(uuid.uuid4())
        temp_file_path = self.temp_dir / f"{file_id}_{filename}"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        return str(temp_file_path)
    
    def cleanup_temp_file(self, file_path: str):
        """清理临时文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"清理临时文件失败 {file_path}: {str(e)}")
    
    def _process_document_sync(self, file_path: str, request: ChunkRequest) -> Dict[str, Any]:
        """同步处理文档（在线程池中执行）"""
        try:
            start_time = time.time()
            
            # 确定解析器类型
            if request.parser_type == "auto":
                parser_type = self.detect_parser_type(file_path)
            else:
                parser_type = request.parser_type
            
            # 创建分块器实例
            logger.info(f"🔧 创建分块器，参数: parser_type={parser_type}, chunk_token_num={request.chunk_token_num}")
            document_chunker = DocumentChunker(
                parser_type=parser_type,
                chunk_token_num=request.chunk_token_num,
                delimiter=request.delimiter,
                language=request.language,
                layout_recognize=request.layout_recognize,
                zoomin=request.zoomin,
                from_page=request.from_page,
                to_page=request.to_page
            )
            
            # 执行分块
            chunks = document_chunker.chunk_document(file_path=file_path)
            processing_time = time.time() - start_time
            
            logger.info(f"🔧 分块完成，生成了 {len(chunks)} 个分块，耗时 {processing_time:.2f}s")
            
            return {
                "success": True,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "processing_time": processing_time,
                "parser_type": parser_type
            }
            
        except Exception as e:
            logger.error(f"处理文档失败 {file_path}: {str(e)}")
            return {
                "success": False,
                "chunks": None,
                "total_chunks": 0,
                "processing_time": time.time() - start_time,
                "parser_type": request.parser_type,
                "error": str(e)
            }
    
    async def process_document(self, file_content: bytes, filename: str, request: ChunkRequest) -> Dict[str, Any]:
        """异步处理单个文档"""
        # 更新统计信息
        stats["total_requests"] += 1
        stats["current_concurrent_tasks"] += 1
        
        async with semaphore:  # 控制并发数
            temp_file_path = None
            try:
                # 保存临时文件
                temp_file_path = self.save_temp_file(file_content, filename)
                
                # 在线程池中执行分块处理
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor,
                    self._process_document_sync,
                    temp_file_path,
                    request
                )
                
                # 添加文件大小信息
                result["file_size"] = len(file_content)
                
                # 更新统计信息
                if result["success"]:
                    stats["successful_requests"] += 1
                else:
                    stats["failed_requests"] += 1
                
                stats["processing_times"].append(result["processing_time"])
                
                return result
                
            finally:
                # 清理临时文件
                if temp_file_path:
                    self.cleanup_temp_file(temp_file_path)
                stats["current_concurrent_tasks"] -= 1
    
    async def process_documents_batch(self, files_data: List[tuple], request: ChunkRequest) -> Dict[str, Any]:
        """异步批量处理文档"""
        start_time = time.time()
        
        # 创建并发任务
        tasks = []
        for file_content, filename in files_data:
            task = self.process_document(file_content, filename, request)
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        successful_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "filename": files_data[i][1],
                    "success": False,
                    "error": str(result),
                    "chunks": None,
                    "total_chunks": 0,
                    "processing_time": 0,
                    "file_size": len(files_data[i][0])
                })
                failed_count += 1
            else:
                processed_results.append({
                    "filename": files_data[i][1],
                    **result
                })
                if result["success"]:
                    successful_count += 1
                else:
                    failed_count += 1
        
        total_processing_time = time.time() - start_time
        
        return {
            "success": failed_count == 0,
            "results": processed_results,
            "total_files": len(files_data),
            "successful_files": successful_count,
            "failed_files": failed_count,
            "total_processing_time": total_processing_time
        }

# 服务实例
chunk_service = ChunkService()

# ==================== HTTP 接口端点 ====================

@app.get("/")
async def root():
    """根路径 - 服务状态"""
    return {
        "service": "DeepRAG 文档分块服务",
        "version": "2.0.0",
        "status": "running",
        "supported_formats": list(Config.SUPPORTED_FORMATS),
        "max_file_size": f"{Config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB",
        "max_concurrent_tasks": Config.MAX_CONCURRENT_TASKS,
        "api_docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - stats["start_time"],
        "current_concurrent_tasks": stats["current_concurrent_tasks"],
        "total_requests": stats["total_requests"]
    }

@app.post("/chunk", response_model=ChunkResponse)
async def chunk_document(
    file: UploadFile = File(...),
    parser_type: str = Form("auto"),
    chunk_token_num: int = Form(256),
    delimiter: str = Form("\n。；！？"),
    language: str = Form("Chinese"),
    layout_recognize: str = Form("DeepDOC"),
    zoomin: int = Form(3),
    from_page: int = Form(0),
    to_page: int = Form(100000)
):
    """
    文档分块接口 - 直接返回分块结果
    
    支持的文件格式: PDF, DOCX, DOC, TXT, MD, HTML, PPTX, XLSX
    """
    # 验证文件
    if not chunk_service.validate_file(file):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持的格式: {', '.join(Config.SUPPORTED_FORMATS)}"
        )
    
    # 读取文件内容
    file_content = await file.read()
    
    # 检查文件大小
    if len(file_content) > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制 ({Config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"
        )
    
    # 创建请求对象
    request = ChunkRequest(
        parser_type=parser_type,
        chunk_token_num=chunk_token_num,
        delimiter=delimiter,
        language=language,
        layout_recognize=layout_recognize,
        zoomin=zoomin,
        from_page=from_page,
        to_page=to_page
    )
    
    # 处理文档
    result = await chunk_service.process_document(file_content, file.filename, request)
    
    # 返回结果
    if result["success"]:
        return ChunkResponse(
            success=True,
            chunks=result["chunks"],
            total_chunks=result["total_chunks"],
            processing_time=result["processing_time"],
            file_size=result["file_size"],
            parser_type=result["parser_type"]
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"文档分块处理失败: {result['error']}"
        )

@app.post("/chunk/batch", response_model=BatchChunkResponse)
async def chunk_documents_batch(
    files: List[UploadFile] = File(...),
    parser_type: str = Form("auto"),
    chunk_token_num: int = Form(256),
    delimiter: str = Form("\n。；！？"),
    language: str = Form("Chinese"),
    layout_recognize: str = Form("DeepDOC"),
    zoomin: int = Form(3),
    from_page: int = Form(0),
    to_page: int = Form(100000)
):
    """
    批量文档分块接口 - 同时处理多个文档
    
    最多支持同时处理 10 个文件
    """
    # 检查文件数量
    if len(files) > Config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"批量上传文件数量不能超过 {Config.MAX_BATCH_SIZE} 个"
        )
    
    # 验证和读取文件
    files_data = []
    for file in files:
        if not chunk_service.validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file.filename} 格式不支持"
            )
        
        file_content = await file.read()
        
        if len(file_content) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件 {file.filename} 大小超过限制"
            )
        
        files_data.append((file_content, file.filename))
    
    # 创建请求对象
    request = ChunkRequest(
        parser_type=parser_type,
        chunk_token_num=chunk_token_num,
        delimiter=delimiter,
        language=language,
        layout_recognize=layout_recognize,
        zoomin=zoomin,
        from_page=from_page,
        to_page=to_page
    )
    
    # 批量处理文档
    result = await chunk_service.process_documents_batch(files_data, request)
    
    return BatchChunkResponse(**result)

@app.get("/stats", response_model=ServiceStats)
async def get_statistics():
    """
    获取服务统计信息
    """
    uptime = time.time() - stats["start_time"]
    avg_processing_time = (
        sum(stats["processing_times"]) / len(stats["processing_times"])
        if stats["processing_times"] else 0
    )
    
    return ServiceStats(
        uptime=uptime,
        total_requests=stats["total_requests"],
        successful_requests=stats["successful_requests"],
        failed_requests=stats["failed_requests"],
        average_processing_time=avg_processing_time,
        current_concurrent_tasks=stats["current_concurrent_tasks"]
    )

@app.post("/admin/cleanup")
async def cleanup_temp_files():
    """
    清理临时文件
    
    删除临时目录中的所有文件
    """
    try:
        temp_dir = Path(Config.TEMP_DIR)
        if temp_dir.exists():
            file_count = len(list(temp_dir.glob("*")))
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(exist_ok=True)
            
            return {
                "message": f"已清理 {file_count} 个临时文件",
                "cleaned_files": file_count
            }
        else:
            return {
                "message": "临时目录不存在",
                "cleaned_files": 0
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清理临时文件失败: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("DeepRAG 文档分块服务启动中...")
    
    # 创建临时目录
    temp_dir = Path(Config.TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)
    
    logger.info(f"服务配置:")
    logger.info(f"  - 最大工作线程数: {Config.MAX_WORKERS}")
    logger.info(f"  - 最大文件大小: {Config.MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
    logger.info(f"  - 最大并发任务数: {Config.MAX_CONCURRENT_TASKS}")
    logger.info(f"  - 最大批量处理数: {Config.MAX_BATCH_SIZE}")
    logger.info(f"  - 支持的文件格式: {', '.join(Config.SUPPORTED_FORMATS)}")
    logger.info(f"  - 临时文件目录: {Config.TEMP_DIR}")
    logger.info("DeepRAG 文档分块服务启动完成!")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    logger.info("DeepRAG 文档分块服务关闭中...")
    
    # 关闭线程池
    executor.shutdown(wait=True)
    
    # 清理临时文件
    temp_dir = Path(Config.TEMP_DIR)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info("DeepRAG 文档分块服务已关闭")

# ==================== 服务启动 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepRAG 文档分块服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8089, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    logger.info(f"启动 DeepRAG 文档分块服务...")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "chunk_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )
