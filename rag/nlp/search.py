#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import re
from dataclasses import dataclass

from rag.settings import TAG_FLD, PAGERANK_FLD
from rag.utils import rmSpace
from rag.nlp import rag_tokenizer, query
import numpy as np
from rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr


def index_name(uid):
    """根据用户ID生成索引名称"""
    return f"ragflow_{uid}"


class Dealer:
    """
    搜索处理器类

    负责处理文档搜索、向量检索、重排序等核心功能
    """
    def __init__(self, dataStore: DocStoreConnection):
        """
        初始化搜索处理器

        Args:
            dataStore: 文档存储连接实例
        """
        self.qryr = query.FulltextQueryer()  # 全文搜索查询器
        self.dataStore = dataStore           # 数据存储连接

    @dataclass
    class SearchResult:
        """
        搜索结果数据类

        封装搜索返回的各种信息
        """
        total: int                                    # 匹配的总文档数
        ids: list[str]                               # 文档分块ID列表
        query_vector: list[float] | None = None      # 查询向量
        field: dict | None = None                    # 字段数据
        highlight: dict | None = None                # 高亮信息
        aggregation: list | dict | None = None       # 聚合结果
        keywords: list[str] | None = None            # 关键词列表
        group_docs: list[list] | None = None         # 分组文档

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """
        将文本转换为向量匹配表达式

        Args:
            txt: 输入文本
            emb_mdl: 嵌入模型
            topk: 返回的top-K结果数量
            similarity: 相似度阈值

        Returns:
            MatchDenseExpr: 密集向量匹配表达式
        """
        qv, _ = emb_mdl.encode_queries(txt)  # 编码查询文本为向量
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]  # 转换为浮点数列表
        vector_column_name = f"q_{len(embedding_data)}_vec"  # 生成向量列名
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        """
        从请求中提取过滤条件

        Args:
            req: 请求参数字典

        Returns:
            dict: 过滤条件字典
        """
        condition = dict()
        # 处理知识库ID和文档ID映射
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]

        # 处理其他过滤字段
        # TODO(yzc): `available_int` 是可空的，但 infinity 不支持可空列
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: dict | None = None
               ):
        """
        执行文档搜索

        Args:
            req: 搜索请求参数
            idx_names: 索引名称（单个或列表）
            kb_ids: 知识库ID列表
            emb_mdl: 嵌入模型（可选）
            highlight: 是否启用高亮
            rank_feature: 排序特征参数

        Returns:
            SearchResult: 搜索结果对象
        """
        filters = self.get_filters(req)  # 获取过滤条件
        orderBy = OrderByExpr()          # 创建排序表达式

        # 解析分页参数
        pg = int(req.get("page", 1)) - 1    # 页码（从0开始）
        topk = int(req.get("topk", 1024))   # 向量搜索的top-K
        ps = int(req.get("size", topk))     # 页面大小
        offset, limit = pg * ps, ps         # 计算偏移量和限制

        # 设置要返回的字段
        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        kwds = set([])  # 关键词集合

        qst = req.get("question", "")  # 获取查询问题
        q_vec = []                     # 查询向量

        if not qst:
            # 如果没有查询问题，执行简单的列表查询
            if req.get("sort"):
                # 设置排序：按页码、位置升序，创建时间降序
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # 有查询问题的情况，执行混合搜索
            highlightFields = ["content_ltks", "title_tks"] if highlight else []  # 高亮字段
            matchText, keywords = self.qryr.question(qst, min_match=0.3)  # 生成文本匹配表达式

            if emb_mdl is None:
                # 仅使用文本搜索
                matchExprs = [matchText]
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 使用文本+向量混合搜索
                matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))  # 生成向量匹配表达式
                q_vec = matchDense.embedding_data  # 保存查询向量
                src.append(f"q_{len(q_vec)}_vec")  # 添加向量字段到返回字段

                # 创建融合表达式：文本权重0.05，向量权重0.95
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 如果结果为空，尝试降低匹配阈值重新搜索
                if total == 0:
                    matchText, _ = self.qryr.question(qst, min_match=0.1)  # 降低文本匹配阈值
                    filters.pop("doc_ids", None)  # 移除文档ID限制
                    matchDense.extra_options["similarity"] = 0.17  # 降低向量相似度阈值
                    res = self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                    total = self.dataStore.getTotal(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            # 处理关键词，包括细粒度分词
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:  # 跳过太短的词
                        continue
                    if kk in kwds:   # 避免重复
                        continue
                    kwds.add(kk)

        # 记录搜索结果统计
        logging.debug(f"TOTAL: {total}")

        # 提取搜索结果的各个组成部分
        ids = self.dataStore.getChunkIds(res)  # 获取分块ID列表
        keywords = list(kwds)                  # 转换关键词集合为列表
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")  # 获取高亮信息
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")  # 获取文档名聚合信息

        # 构建并返回搜索结果对象
        return self.SearchResult(
            total=total,                                    # 总匹配数量
            ids=ids,                                       # 分块ID列表
            query_vector=q_vec,                            # 查询向量
            aggregation=aggs,                              # 聚合结果
            highlight=highlight,                           # 高亮信息
            field=self.dataStore.getFields(res, src),      # 字段数据
            keywords=keywords                              # 关键词列表
        )

    # @staticmethod
    # def trans2floats(txt):
    #     """
    #     将制表符分隔的字符串转换为浮点数列表
    #
    #     Args:
    #         txt: 制表符分隔的数字字符串
    #
    #     Returns:
    #         list[float]: 浮点数列表
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     return [float(t) for t in txt.split("\t")]

    # def insert_citations(self, answer, chunks, chunk_v,
    #                      embd_mdl, tkweight=0.1, vtweight=0.9):
    #     """
    #     在答案中插入引用标记
    #
    #     Args:
    #         answer: 原始答案文本
    #         chunks: 文档分块列表
    #         chunk_v: 分块向量列表
    #         embd_mdl: 嵌入模型
    #         tkweight: 词汇相似度权重
    #         vtweight: 向量相似度权重
    #
    #     Returns:
    #         tuple: (带引用的答案, 引用的分块ID集合)
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     assert len(chunks) == len(chunk_v)  # 确保分块和向量数量一致
    #     if not chunks:
    #         return answer, set([])

    #     # 处理代码块，避免在代码块内部分割
    #     pieces = re.split(r"(```)", answer)
    #     if len(pieces) >= 3:
    #         i = 0
    #         pieces_ = []
    #         while i < len(pieces):
    #             if pieces[i] == "```":
    #                 # 找到代码块的开始和结束
    #                 st = i
    #                 i += 1
    #                 while i < len(pieces) and pieces[i] != "```":
    #                     i += 1
    #                 if i < len(pieces):
    #                     i += 1
    #                 pieces_.append("".join(pieces[st: i]) + "\n")
    #             else:
    #                 # 对非代码块部分按句子分割
    #                 pieces_.extend(
    #                     re.split(
    #                         r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
    #                         pieces[i]))
    #                 i += 1
    #         pieces = pieces_
    #     else:
    #         # 没有代码块，直接按句子分割
    #         pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
    #     # 合并标点符号到前一个片段
    #     for i in range(1, len(pieces)):
    #         if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
    #             pieces[i - 1] += pieces[i][0]
    #             pieces[i] = pieces[i][1:]
    #
    #     # 过滤掉太短的片段
    #     idx = []
    #     pieces_ = []
    #     for i, t in enumerate(pieces):
    #         if len(t) < 5:  # 跳过太短的片段
    #             continue
    #         idx.append(i)
    #         pieces_.append(t)
    #
    #     logging.debug("{} => {}".format(answer, pieces_))
    #     if not pieces_:
    #         return answer, set([])
    #
    #     # 对答案片段进行向量编码
    #     ans_v, _ = embd_mdl.encode(pieces_)
    #
    #     # 确保向量维度一致
    #     for i in range(len(chunk_v)):
    #         if len(ans_v[0]) != len(chunk_v[i]):
    #             chunk_v[i] = [0.0]*len(ans_v[0])
    #             logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))
    #
    #     assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
    #         len(ans_v[0]), len(chunk_v[0]))
    #
    #     # 对分块内容进行分词
    #     chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
    #                   for ck in chunks]
    #
    #     # 寻找引用匹配，使用递减的阈值
    #     cites = {}
    #     thr = 0.63  # 初始相似度阈值
    #     while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
    #         for i, _ in enumerate(pieces_):  # 遍历答案片段
    #             # 计算混合相似度（词汇+向量）
    #             sim, _, _ = self.qryr.hybrid_similarity(ans_v[i],
    #                                                             chunk_v,
    #                                                             rag_tokenizer.tokenize(
    #                                                                 self.qryr.rmWWW(pieces_[i])).split(),
    #                                                             chunks_tks,
    #                                                             tkweight, vtweight)
    #             mx = np.max(sim) * 0.99  # 最大相似度的99%作为阈值
    #             logging.debug("{} SIM: {}".format(pieces_[i], mx))
    #             if mx < thr:
    #                 continue
    #             # 找到相似度超过阈值的分块，最多4个
    #             cites[idx[i]] = list(
    #                 set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
    #         thr *= 0.8  # 降低阈值重试
    #
    #     # 重新组装答案，插入引用标记
    #     res = ""
    #     seted = set([])  # 已使用的引用ID集合
    #     for i, p in enumerate(pieces):
    #         res += p
    #         if i not in idx:  # 跳过被过滤的片段
    #             continue
    #         if i not in cites:  # 跳过没有引用的片段
    #             continue
    #
    #         # 验证引用ID的有效性
    #         for c in cites[i]:
    #             assert int(c) < len(chunk_v)
    #
    #         # 添加引用标记，避免重复
    #         for c in cites[i]:
    #             if c in seted:
    #                 continue
    #             res += f" ##{c}$$"  # 引用标记格式
    #             seted.add(c)
    #
    #     return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        """
        计算排序特征评分

        Args:
            query_rfea: 查询的排序特征
            search_res: 搜索结果对象

        Returns:
            np.array: 排序特征评分数组
        """
        # 计算排序特征（标签特征）评分
        rank_fea = []
        pageranks = []

        # 提取每个分块的 PageRank 分数
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            # 如果没有查询特征，返回零向量加上 PageRank
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        # 计算查询特征的归一化因子
        q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))

        # 计算每个分块的特征匹配分数
        for i in search_res.ids:
            nor, denor = 0, 0
            # 解析分块的标签特征
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc  # 计算点积
                denor += sc * sc  # 计算分块特征的模长平方

            if denor == 0:
                rank_fea.append(0)
            else:
                # 计算余弦相似度
                rank_fea.append(nor/np.sqrt(denor)/q_denor)

        # 返回特征分数（放大10倍）加上 PageRank 分数
        return np.array(rank_fea)*10. + pageranks

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        """
        对搜索结果进行重排序

        Args:
            sres: 搜索结果对象
            query: 查询文本
            tkweight: 词汇相似度权重
            vtweight: 向量相似度权重
            cfield: 内容字段名
            rank_feature: 排序特征参数

        Returns:
            tuple: (综合相似度, 词汇相似度, 向量相似度)
        """
        _, keywords = self.qryr.question(query)  # 提取查询关键词
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size

        # 提取分块向量
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                # 如果向量是字符串格式，转换为浮点数列表
                vector = [float(v) for v in vector.split("\t")]
            ins_embd.append(vector)

        if not ins_embd:
            return [], [], []

        # 确保重要关键词字段是列表格式
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]

        # 构建分块的词汇表示
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()  # 内容词汇
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]  # 标题词汇
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]  # 问题词汇
            important_kwd = sres.field[i].get("important_kwd", [])  # 重要关键词

            # 组合词汇，不同类型给予不同权重
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # 计算排序特征评分
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # 计算混合相似度
        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim

    # def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
    #     """
    #     计算混合相似度的简化接口
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     return self.qryr.hybrid_similarity(ans_embd,
    #                                        ins_embd,
    #                                        rag_tokenizer.tokenize(ans).split(),
    #                                        rag_tokenizer.tokenize(inst).split())

    def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: dict | None = {PAGERANK_FLD: 10}):
        """
        执行文档检索

        Args:
            question: 查询问题
            embd_mdl: 嵌入模型
            tenant_ids: 租户ID列表
            kb_ids: 知识库ID列表
            page: 页码
            page_size: 页面大小
            similarity_threshold: 相似度阈值
            vector_similarity_weight: 向量相似度权重
            top: 向量搜索的top-K
            doc_ids: 文档ID列表（可选）
            aggs: 是否计算聚合
            rerank_mdl: 重排序模型（可选）
            highlight: 是否启用高亮
            rank_feature: 排序特征参数

        Returns:
            dict: 检索结果，包含总数、分块列表和文档聚合信息
        """
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        RERANK_PAGE_LIMIT = 3  # 重排序页面限制

        # 构建搜索请求
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "size": max(page_size * RERANK_PAGE_LIMIT, 128),
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}

        # 如果页码超过重排序限制，直接使用原始分页
        if page > RERANK_PAGE_LIMIT:
            req["page"] = page
            req["size"] = page_size

        # 处理租户ID
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        # 执行搜索
        sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                           kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
        ranks["total"] = sres.total

        if page <= RERANK_PAGE_LIMIT:
            if rerank_mdl and sres.total > 0:
                sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                                       sres, question, 1 - vector_similarity_weight,
                                                       vector_similarity_weight,
                                                       rank_feature=rank_feature)
            else:
                sim, tsim, vsim = self.rerank(
                    sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                    rank_feature=rank_feature)
            idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]
        else:
            sim = tsim = vsim = [1] * len(sres.ids)
            idx = list(range(len(sres.ids)))

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")
            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                                                       v in sorted(ranks["doc_aggs"].items(),
                                                                   key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks

    # def sql_retrieval(self, sql, fetch_size=128, format="json"):
    #     """
    #     SQL查询检索
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     tbl = self.dataStore.sql(sql, fetch_size, format)
    #     return tbl

    # def chunk_list(self, doc_id: str, tenant_id: str,
    #                kb_ids: list[str], max_count=1024,
    #                offset=0,
    #                fields=["docnm_kwd", "content_with_weight", "img_id"]):
    #     """
    #     获取分块列表
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     condition = {"doc_id": doc_id}
    #     res = []
    #     bs = 128
    #     for p in range(offset, max_count, bs):
    #         es_res = self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id),
    #                                        kb_ids)
    #         dict_chunks = self.dataStore.getFields(es_res, fields)
    #         for id, doc in dict_chunks.items():
    #             doc["id"] = id
    #         if dict_chunks:
    #             res.extend(dict_chunks.values())
    #         if len(dict_chunks.values()) < bs:
    #             break
    #     return res

    # def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
    #     """
    #     获取所有标签
    #
    #     Args:
    #         tenant_id: 租户ID
    #         kb_ids: 知识库ID列表
    #         S: 平滑参数（保留用于接口兼容性，当前未使用）
    #
    #     Returns:
    #         list: 标签聚合结果
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     _ = S  # 标记参数已知但未使用
    #     if not self.dataStore.indexExist(index_name(tenant_id), kb_ids[0]):
    #         return []
    #     res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
    #     return self.dataStore.getAggregation(res, "tag_kwd")

    # def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
    #     """
    #     获取标签的比例分布
    #
    #     Args:
    #         tenant_id: 租户ID
    #         kb_ids: 知识库ID列表
    #         S: 平滑参数
    #
    #     Returns:
    #         dict: 标签及其比例的字典
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
    #     res = self.dataStore.getAggregation(res, "tag_kwd")
    #     total = np.sum([c for _, c in res])  # 计算总数
    #     return {t: (c + 1) / (total + S) for t, c in res}  # 返回平滑后的比例

    # def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
    #     """
    #     为文档内容打标签
    #
    #     Args:
    #         tenant_id: 租户ID
    #         kb_ids: 知识库ID列表
    #         doc: 文档对象
    #         all_tags: 所有标签的分布
    #         topn_tags: 返回的top-N标签数量
    #         keywords_topn: 关键词top-N数量
    #         S: 平滑参数
    #
    #     Returns:
    #         bool: 是否成功打标签
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     idx_nm = index_name(tenant_id)
    #     # 构建段落匹配表达式
    #     match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
    #     res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
    #     aggs = self.dataStore.getAggregation(res, "tag_kwd")
    #     if not aggs:
    #         return False
    #
    #     cnt = np.sum([c for _, c in aggs])  # 计算总数
    #     # 计算标签特征分数：TF-IDF 风格的计算
    #     tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
    #                      key=lambda x: x[1] * -1)[:topn_tags]
    #     doc[TAG_FLD] = {a: c for a, c in tag_fea if c > 0}  # 保存标签特征
    #     return True

    # def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
    #     """
    #     为查询问题提取相关标签
    #
    #     Args:
    #         question: 查询问题
    #         tenant_ids: 租户ID（单个或列表）
    #         kb_ids: 知识库ID列表
    #         all_tags: 所有标签的分布
    #         topn_tags: 返回的top-N标签数量
    #         S: 平滑参数
    #
    #     Returns:
    #         dict: 标签及其权重的字典
    #
    #     注意：此方法当前未被使用，已注释
    #     """
    #     if isinstance(tenant_ids, str):
    #         idx_nms = index_name(tenant_ids)
    #     else:
    #         idx_nms = [index_name(tid) for tid in tenant_ids]
    #
    #     match_txt, _ = self.qryr.question(question, min_match=0.0)  # 生成问题匹配表达式
    #     res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
    #     aggs = self.dataStore.getAggregation(res, "tag_kwd")
    #     if not aggs:
    #         return {}
    #
    #     cnt = np.sum([c for _, c in aggs])  # 计算总数
    #     # 计算标签特征分数：TF-IDF 风格的计算
    #     tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
    #                      key=lambda x: x[1] * -1)[:topn_tags]
    #     return {a: max(1, c) for a, c in tag_fea}  # 返回标签权重，最小值为1
