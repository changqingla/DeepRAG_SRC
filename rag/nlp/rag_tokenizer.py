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
import copy
import datrie
import math
import os
import re
import string
import sys
from hanziconv import HanziConv
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from rag.utils import get_project_base_directory


class RagTokenizer:
    def key_(self, line):
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def loadDict_(self, fnm, save_cache=True):
        logging.info(f"[HUQIE]:Build trie from {fnm}")
        try:
            of = open(fnm, "r", encoding='utf-8')
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1

            if save_cache:
                dict_file_cache = fnm + ".trie"
                logging.info(f"[HUQIE]:Build trie cache to {dict_file_cache}")
                self.trie_.save(dict_file_cache)
            of.close()
        except Exception:
            logging.exception(f"[HUQIE]:Build trie {fnm} failed")

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.DENOMINATOR = 1000000
        self.DIR_ = os.path.join(get_project_base_directory(), "rag/res", "huqie")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.SPLIT_CHAR = r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；''：""【】~！￥%……（）——-]+|[a-zA-Z0-9,\.-]+)"

        trie_file_name = self.DIR_ + ".txt.trie"
        base_dict_path = self.DIR_ + ".txt"
        custom_dict_path = os.path.join(get_project_base_directory(), "rag/res", "custom_dict.txt")
        
        # 检查是否需要重建trie缓存
        need_rebuild = self._need_rebuild_trie(trie_file_name, base_dict_path, custom_dict_path)
        
        if need_rebuild:
            logging.info(f"[HUQIE]:Building trie cache (base dict updated or custom dict changed)")
            self.trie_ = datrie.Trie(string.printable)
            # 加载基础词典，但不保存缓存
            self.loadDict_(base_dict_path, save_cache=False)
            # 加载自定义词典
            if os.path.exists(custom_dict_path):
                logging.info(f"[HUQIE]:Loading custom dictionary from {custom_dict_path}")
                self.addUserDict(custom_dict_path)
                logging.info(f"[HUQIE]:Custom dictionary loaded successfully")
            # 保存包含所有词典的完整trie缓存
            logging.info(f"[HUQIE]:Saving complete trie cache to {trie_file_name}")
            self.trie_.save(trie_file_name)
        else:
            try:
                # 从缓存加载trie
                logging.info(f"[HUQIE]:Loading trie from cache {trie_file_name}")
                self.trie_ = datrie.Trie.load(trie_file_name)
                logging.info(f"[HUQIE]:Trie cache loaded successfully")
            except Exception:
                # 加载失败，重建trie
                logging.exception(f"[HUQIE]:Fail to load trie file {trie_file_name}, rebuilding trie")
                self.trie_ = datrie.Trie(string.printable)
                self.loadDict_(base_dict_path, save_cache=False)
                if os.path.exists(custom_dict_path):
                    logging.info(f"[HUQIE]:Loading custom dictionary from {custom_dict_path}")
                    self.addUserDict(custom_dict_path)
                    logging.info(f"[HUQIE]:Custom dictionary loaded successfully")
                # 保存包含所有词典的完整trie缓存
                logging.info(f"[HUQIE]:Saving complete trie cache to {trie_file_name}")
                self.trie_.save(trie_file_name)

    def _need_rebuild_trie(self, trie_file_name, base_dict_path, custom_dict_path):
        """检查是否需要重建trie缓存"""
        # 如果trie文件不存在，需要重建
        if not os.path.exists(trie_file_name):
            logging.info(f"[HUQIE]:Trie file {trie_file_name} not found, need to build")
            return True
        
        try:
            trie_mtime = os.path.getmtime(trie_file_name)
            
            # 检查基础词典是否更新
            if os.path.exists(base_dict_path):
                base_dict_mtime = os.path.getmtime(base_dict_path)
                if base_dict_mtime > trie_mtime:
                    logging.info(f"[HUQIE]:Base dictionary {base_dict_path} updated, need to rebuild trie")
                    return True
            
            # 检查自定义词典是否更新
            if os.path.exists(custom_dict_path):
                custom_dict_mtime = os.path.getmtime(custom_dict_path)
                if custom_dict_mtime > trie_mtime:
                    logging.info(f"[HUQIE]:Custom dictionary {custom_dict_path} updated, need to rebuild trie")
                    return True
            
            # 所有文件都没有更新，不需要重建
            return False
            
        except OSError as e:
            logging.warning(f"[HUQIE]:Failed to check file modification time: {e}, rebuilding trie")
            return True

    def loadUserDict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception:
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm, save_cache=False)

    def _strQ2B(self, ustring):
        """Convert full-width characters to half-width characters"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # After the conversion, if it's not a half-width character, return the original character.
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        if s + 2 <= len(chars):
            t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
                    self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    pretks.append((t, self.trie_[k]))
                else:
                    pretks.append((t, (-12, '')))
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        #F /= len(tks)
        L /= len(tks)
        logging.debug("[SC] {} {} {} {} {}".format(tks, len(tks), L, F, B / len(tks) + L + F))
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split()
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def maxForward_(self, line):
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in tks]

    def _split_by_lang(self, line):
        txt_lang_pairs = []
        arr = re.split(self.SPLIT_CHAR, line)
        for a in arr:
            if not a:
                continue
            s = 0
            e = s + 1
            zh = is_chinese(a[s])
            while e < len(a):
                _zh = is_chinese(a[e])
                if _zh == zh:
                    e += 1
                    continue
                txt_lang_pairs.append((a[s: e], zh))
                s = e
                e = s + 1
                zh = _zh
            if s >= len(a):
                continue
            txt_lang_pairs.append((a[s: e], zh))
        return txt_lang_pairs

    def tokenize(self, line):
        line = re.sub(r"\W+", " ", line)
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)

        arr = self._split_by_lang(line)
        res = []
        for L,lang in arr:
            if not lang:
                res.extend([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(L)])
                continue
            if len(L) < 2 or re.match(
                    r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
                res.append(L)
                continue

            # use maxforward for the first time
            tks, s = self.maxForward_(L)
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                logging.debug("[FW] {} {}".format(tks, s))
                logging.debug("[BW] {} {}".format(tks1, s1))

            i, j, _i, _j = 0, 0, 0, 0
            same = 0
            while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
                same += 1
            if same > 0:
                res.append(" ".join(tks[j: j + same]))
            _i = i + same
            _j = j + same
            j = _j + 1
            i = _i + 1

            while i < len(tks1) and j < len(tks):
                tk1, tk = "".join(tks1[_i:i]), "".join(tks[_j:j])
                if tk1 != tk:
                    if len(tk1) > len(tk):
                        j += 1
                    else:
                        i += 1
                    continue

                if tks1[i] != tks[j]:
                    i += 1
                    j += 1
                    continue
                # backward tokens from_i to i are different from forward tokens from _j to j.
                tkslist = []
                self.dfs_("".join(tks[_j:j]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                same = 1
                while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
                    same += 1
                res.append(" ".join(tks[j: j + same]))
                _i = i + same
                _j = j + same
                j = _j + 1
                i = _i + 1

            if _i < len(tks1):
                assert _j < len(tks)
                assert "".join(tks1[_i:]) == "".join(tks[_j:])
                tkslist = []
                self.dfs_("".join(tks[_j:]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

        res = " ".join(res)
        logging.debug("[TKS] {}".format(self.merge_(res)))
        return self.merge_(res)

    def fine_grained_tokenize(self, tks):
        tks = tks.split()
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(self.english_normalize_(res))


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split():
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
loadUserDict = tokenizer.loadUserDict
addUserDict = tokenizer.addUserDict
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

if __name__ == '__main__':
    tknzr = RagTokenizer(debug=True)
    # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    tks = tknzr.tokenize(
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("虽然我不怎么玩")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("这周日你去吗？这周日你有空吗？")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    logging.info(tknzr.fine_grained_tokenize(tks))
    if len(sys.argv) < 2:
        sys.exit()
    tknzr.DEBUG = False
    tknzr.loadUserDict(sys.argv[1])
    of = open(sys.argv[2], "r")
    while True:
        line = of.readline()
        if not line:
            break
        logging.info(tknzr.tokenize(line))
    of.close()
