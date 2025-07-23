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

"""
查询处理模块

该模块提供智能查询处理功能，主要用于：
1. 全文搜索查询构建和优化
2. 查询重写和扩展（同义词、细粒度分词）
3. 混合相似度计算（向量+文本）
4. 段落级查询生成

核心功能：
- 智能查询预处理（去噪、标准化、语言检测）
- 多字段权重查询构建（标题、内容、关键词等）
- 同义词扩展和细粒度分词
- 文本相似度计算和混合检索
- 支持中英文混合查询处理

主要用于RAG系统中的查询理解、检索优化、相关性计算等场景。

作者: InfiniFlow Authors
许可证: Apache 2.0
"""

import logging
import json
import re
from rag.utils.doc_store_conn import MatchTextExpr

from rag.nlp import rag_tokenizer, term_weight
# from rag.nlp import synonym  # 暂时注释掉，避免导入问题


class FulltextQueryer:
    """
    全文搜索查询器类

    负责处理用户查询，构建优化的Elasticsearch查询语句。

    主要功能：
    1. 查询预处理和标准化
    2. 多字段权重查询构建
    3. 同义词扩展和查询重写
    4. 文本相似度计算
    5. 混合检索（向量+文本）

    查询字段权重体系：
    - important_kwd: 重要关键词（权重30）
    - question_tks: 问题分词（权重20）
    - important_tks: 重要分词（权重20）
    - title_tks: 标题分词（权重10）
    - title_sm_tks: 标题细粒度分词（权重5）
    - content_ltks: 内容分词（权重2）
    - content_sm_ltks: 内容细粒度分词（权重1）
    """

    def __init__(self):
        """
        初始化全文搜索查询器

        加载必要的组件：
        1. 词汇权重计算器 - 用于计算词汇重要性
        2. 同义词处理器 - 用于查询扩展
        3. 查询字段配置 - 定义搜索字段和权重
        """
        # 词汇权重计算器，用于计算词汇重要性
        self.tw = term_weight.Dealer()

        # 同义词处理器，用于查询扩展（暂时禁用）
        # self.syn = synonym.Dealer()
        self.syn = None  # 暂时禁用同义词功能

        # 查询字段配置，按权重从高到低排列
        # 格式：字段名^权重值
        self.query_fields = [
            "title_tks^10",        # 标题分词，权重10
            "title_sm_tks^5",      # 标题细粒度分词，权重5
            "important_kwd^30",    # 重要关键词，权重30（最高）
            "important_tks^20",    # 重要分词，权重20
            "question_tks^20",     # 问题分词，权重20
            "content_ltks^2",      # 内容分词，权重2
            "content_sm_ltks",     # 内容细粒度分词，权重1（默认）
        ]

    @staticmethod
    def subSpecialChar(line):
        """
        转义Elasticsearch查询中的特殊字符

        对查询字符串中的特殊字符进行转义，避免ES查询语法错误。

        Args:
            line (str): 输入字符串

        Returns:
            str: 转义后的字符串

        转义的特殊字符：: { } / [ ] - * " ( ) | + ~ ^
        """
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
    def isChinese(line):
        """
        判断文本是否为中文

        通过统计非英文词汇的比例来判断文本语言类型。

        Args:
            line (str): 输入文本

        Returns:
            bool: True表示中文，False表示英文

        判断逻辑：
        1. 如果词汇数量<=3，默认为中文
        2. 统计非纯英文词汇的比例
        3. 如果非英文词汇比例>=70%，判定为中文
        """
        # 按空白字符分割文本
        arr = re.split(r"[ \t]+", line)

        # 短文本默认为中文
        if len(arr) <= 3:
            return True

        # 统计非纯英文词汇数量
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):  # 不是纯英文字母
                e += 1

        # 计算非英文词汇比例
        return e * 1.0 / len(arr) >= 0.7

    @staticmethod
    def rmWWW(txt):
        """
        移除查询中的无意义词汇（疑问词、助词、介词等）

        通过正则表达式过滤掉对搜索无帮助的词汇，提高查询精度。

        Args:
            txt (str): 输入查询文本

        Returns:
            str: 清理后的查询文本

        过滤的词汇类型：
        1. 中文疑问词：什么、怎么、哪里、为什么等
        2. 英文疑问词：what、who、how、which、where、why等
        3. 英文助词：is、are、the、a、an、of、to等
        """
        # 定义过滤规则
        patts = [
            # 中文疑问词和语气词过滤
            (
                r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            # 英文疑问词过滤
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            # 英文助词、介词、代词等过滤
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ")
        ]

        # 保存原始文本
        otxt = txt

        # 应用过滤规则（忽略大小写）
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)

        # 如果过滤后为空，返回原始文本
        if not txt:
            txt = otxt

        return txt

    def question(self, txt, min_match: float = 0.6):
        """
        构建智能问答查询表达式

        将用户的自然语言问题转换为优化的Elasticsearch查询，支持：
        1. 文本标准化和清理
        2. 语言检测和分支处理
        3. 同义词扩展和权重计算
        4. 细粒度分词和短语匹配
        5. 多字段权重查询构建

        Args:
            txt (str): 用户输入的问题文本
            tbl (str): 表名（兼容参数，当前未使用）
            min_match (float): 最小匹配度，默认0.6（60%的词汇需要匹配）

        Returns:
            tuple: (MatchTextExpr对象, keywords列表) 或 (None, keywords列表)

        处理流程：
        1. 文本预处理：繁简转换、全半角转换、小写化、标点清理
        2. 移除无意义词汇
        3. 语言检测：中文和英文采用不同的处理策略
        4. 英文处理：权重计算 + 同义词扩展 + 短语匹配
        5. 中文处理：细粒度分词 + 同义词扩展 + 复合查询
        """
        # 兼容参数，当前未使用
        # _ = tbl

        # 第一步：文本标准化和清理
        # 1. 转换为简体中文
        # 2. 全角转半角
        # 3. 转换为小写
        # 4. 清理标点符号和特殊字符
        txt = re.sub(
            r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()

        # 第二步：移除无意义词汇（疑问词、助词等）
        txt = FulltextQueryer.rmWWW(txt)

        # 第三步：语言检测和分支处理
        if not self.isChinese(txt):
            # 英文处理分支
            # 再次清理无意义词汇（英文查询需要更彻底的清理）
            txt = FulltextQueryer.rmWWW(txt)

            # 英文分词处理
            tks = rag_tokenizer.tokenize(txt).split()
            keywords = [t for t in tks if t]  # 保存原始关键词

            # 计算词汇权重
            tks_w = self.tw.weights(tks, preprocess=False)

            # 清理词汇：移除特殊字符和无效词汇
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]  # 移除引号等
            tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]  # 移除单字符
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]  # 移除符号前缀
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]  # 去除空白

            # 同义词扩展处理
            syns = []
            for tk, w in tks_w[:256]:  # 限制处理词汇数量
                # 查找同义词（如果同义词功能可用）
                syn = self.syn.lookup(tk) if self.syn else []
                syn = rag_tokenizer.tokenize(" ".join(syn)).split()
                keywords.extend(syn)  # 添加到关键词列表

                # 构建同义词查询片段（权重降低为1/4）
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                syns.append(" ".join(syn))

            # 构建英文查询表达式
            # 1. 单词查询：原词 + 同义词组合
            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if
                 tk and not re.match(r"[.^+\(\)-]", tk)]  # 过滤特殊字符开头的词

            # 2. 短语查询：相邻词汇组合（权重加倍）
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                # 构建短语查询："词1 词2"^权重
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,  # 短语权重为单词权重的2倍
                    )
                )

            # 3. 兜底查询：如果没有构建出查询，使用原始文本
            if not q:
                q.append(txt)

            # 4. 组合最终查询
            query = " ".join(q)
            return MatchTextExpr(
                self.query_fields, query, 100
            ), keywords

        # 中文处理分支
        def need_fine_grained_tokenize(tk):
            """
            判断词汇是否需要细粒度分词

            Args:
                tk (str): 词汇

            Returns:
                bool: True表示需要细粒度分词

            判断条件：
            1. 长度>=3个字符
            2. 不是纯数字、英文、符号组合
            """
            if len(tk) < 3:
                return False  # 短词汇不需要细分
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False  # 数字、英文、符号组合不需要细分
            return True

        # 再次清理文本
        txt = FulltextQueryer.rmWWW(txt)
        qs, keywords = [], []  # 查询片段列表和关键词列表

        # 按词汇分割处理（限制256个词汇）
        for tt in self.tw.split(txt)[:256]:
            if not tt:
                continue

            # 添加到关键词列表
            keywords.append(tt)

            # 计算词汇权重
            twts = self.tw.weights([tt])

            # 查找同义词并添加到关键词（限制32个关键词）
            syns = self.syn.lookup(tt) if self.syn else []
            if syns and len(keywords) < 32:
                keywords.extend(syns)

            # 调试日志：输出权重信息
            logging.debug(json.dumps(twts, ensure_ascii=False))

            # 构建词汇查询片段
            tms = []
            # 按权重降序处理每个词汇
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                # 细粒度分词处理
                sm = (
                    rag_tokenizer.fine_grained_tokenize(tk).split()
                    if need_fine_grained_tokenize(tk)
                    else []
                )
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [FulltextQueryer.subSpecialChar(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                # 同义词处理
                tk_syns = self.syn.lookup(tk) if self.syn else []  # 查找同义词
                tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]  # 转义特殊字符

                # 添加同义词到关键词列表（限制32个）
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])

                # 对同义词进行细粒度分词
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                # 包含空格的同义词用引号包围（短语查询）
                tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]

                # 关键词数量达到限制，停止处理
                if len(keywords) >= 32:
                    break

                # 构建当前词汇的查询表达式
                tk = FulltextQueryer.subSpecialChar(tk)  # 转义原词特殊字符

                # 包含空格的词汇用引号包围（短语查询）
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk

                # 添加同义词查询（权重0.2）
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)

                # 添加细粒度分词查询（精确匹配和邻近匹配，权重0.5）
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))

                # 添加到查询片段列表
                if tk.strip():
                    tms.append((tk, w))

            # 组合当前词汇的所有查询片段
            tms = " ".join([f"({t})^{w}" for t, w in tms])

            # 如果有多个子词汇，添加邻近查询（权重1.5）
            if len(twts) > 1:
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            # 构建同义词查询片段
            syns = " OR ".join(
                [
                    '"%s"'
                    % rag_tokenizer.tokenize(FulltextQueryer.subSpecialChar(s))
                    for s in syns
                ]
            )

            # 组合原词查询和同义词查询（原词权重5，同义词权重0.7）
            if syns and tms:
                tms = f"({tms})^5 OR ({syns})^0.7"

            # 添加到查询片段列表
            qs.append(tms)

        # 构建最终查询表达式
        if qs:
            # 用OR连接所有查询片段
            query = " OR ".join([f"({t})" for t in qs if t])
            return MatchTextExpr(
                self.query_fields, query, 100, {"minimum_should_match": min_match}
            ), keywords

        # 如果没有构建出查询，返回None
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        """
        计算混合相似度（向量相似度 + 文本相似度）

        结合向量相似度和文本相似度，提供更准确的相关性评分。

        Args:
            avec: 查询向量
            bvecs: 文档向量列表
            atks: 查询词汇列表
            btkss: 文档词汇列表的列表
            tkweight (float): 文本相似度权重，默认0.3（30%）
            vtweight (float): 向量相似度权重，默认0.7（70%）

        Returns:
            tuple: (混合相似度数组, 文本相似度数组, 向量相似度数组)

        计算公式：
        混合相似度 = 向量相似度 × 0.7 + 文本相似度 × 0.3
        """
        from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
        import numpy as np

        # 计算向量余弦相似度
        sims = CosineSimilarity([avec], bvecs)

        # 计算文本相似度
        tksim = self.token_similarity(atks, btkss)

        # 如果向量相似度全为0，只返回文本相似度
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]

        # 返回加权混合相似度
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        """
        计算文本相似度

        基于词汇权重计算查询和文档之间的文本相似度。

        Args:
            atks: 查询词汇列表
            btkss: 文档词汇列表的列表

        Returns:
            list: 每个文档与查询的相似度列表

        计算方法：
        1. 将词汇列表转换为权重字典
        2. 计算每个文档与查询的相似度
        """
        def toDict(tks):
            """
            将词汇列表转换为权重字典

            Args:
                tks: 词汇列表或字符串

            Returns:
                dict: {词汇: 权重} 字典
            """
            d = {}
            # 如果输入是字符串，先分词
            if isinstance(tks, str):
                tks = tks.split()

            # 计算每个词汇的权重并累加
            for t, c in self.tw.weights(tks, preprocess=False):
                if t not in d:
                    d[t] = 0
                d[t] += c  # 累加权重（处理重复词汇）
            return d

        # 转换查询词汇为权重字典
        atks = toDict(atks)

        # 转换所有文档词汇为权重字典列表
        btkss = [toDict(tks) for tks in btkss]

        # 计算每个文档与查询的相似度
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        """
        计算两个文本的相似度

        基于词汇权重计算两个文本之间的相似度。

        Args:
            qtwt: 查询权重字典或字符串
            dtwt: 文档权重字典或字符串

        Returns:
            float: 相似度分数（0-1之间）

        计算方法：
        相似度 = 共同词汇权重之和 / 查询词汇权重总和

        这是一种基于查询覆盖度的相似度计算方法。
        """
        # 如果输入是字符串，转换为权重字典
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}

        # 计算共同词汇的权重之和
        s = 1e-9  # 添加小值避免除零错误
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # 只累加查询词汇的权重（注释：原本可能想乘以文档权重）

        # 计算查询词汇的总权重
        q = 1e-9  # 添加小值避免除零错误
        for k, v in qtwt.items():
            q += v

        # 返回相似度：共同权重 / 查询总权重
        return s / q

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        """
        构建段落级查询表达式

        基于段落内容生成查询表达式，用于段落级别的相似度匹配。

        Args:
            content_tks (str): 段落内容词汇（字符串或列表）
            keywords (list): 额外的关键词列表，默认为空
            keywords_topn (int): 提取的关键词数量上限，默认30

        Returns:
            MatchTextExpr: 段落查询表达式对象

        处理流程：
        1. 内容分词和权重计算
        2. 按权重排序选择top关键词
        3. 同义词扩展和查询构建
        4. 设置动态最小匹配度
        """
        # 处理输入内容：如果是字符串则转换为词汇列表
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]

        # 计算内容词汇的权重
        tks_w = self.tw.weights(content_tks, preprocess=False)

        # 预处理已有关键词：用引号包围（短语查询）
        keywords = [f'"{k.strip()}"' for k in keywords]

        # 按权重降序处理top关键词
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            # 查找同义词
            tk_syns = self.syn.lookup(tk) if self.syn else []
            # 转义同义词中的特殊字符
            tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]
            # 对同义词进行细粒度分词
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            # 包含空格的同义词用引号包围
            tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]

            # 处理原词汇
            tk = FulltextQueryer.subSpecialChar(tk)  # 转义特殊字符
            if tk.find(" ") > 0:
                tk = '"%s"' % tk  # 包含空格的词汇用引号包围

            # 添加同义词查询（权重0.2）
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)

            # 添加到关键词列表（带权重）
            if tk:
                keywords.append(f"{tk}^{w}")

        # 构建段落查询表达式
        # 动态设置最小匹配度：最少3个词汇或10%的词汇数量
        return MatchTextExpr(self.query_fields, " ".join(keywords), 100,
                             {"minimum_should_match": min(3, len(keywords) / 10)})
