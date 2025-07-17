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

import os
import re
import tiktoken
from pathlib import Path
from enum import Enum, IntEnum
from strenum import StrEnum


# Enum definitions (replaces api.db enums)
class StatusEnum(Enum):
    VALID = "1"
    INVALID = "0"


class UserTenantRole(StrEnum):
    OWNER = 'owner'
    ADMIN = 'admin'
    NORMAL = 'normal'
    INVITE = 'invite'


class TenantPermission(StrEnum):
    ME = 'me'
    TEAM = 'team'


class SerializedType(IntEnum):
    PICKLE = 1
    JSON = 2


class FileType(StrEnum):
    PDF = 'pdf'
    DOC = 'doc'
    VISUAL = 'visual'
    AURAL = 'aural'
    VIRTUAL = 'virtual'
    FOLDER = 'folder'
    OTHER = "other"


class LLMType(StrEnum):
    CHAT = 'chat'
    EMBEDDING = 'embedding'
    SPEECH2TEXT = 'speech2text'
    IMAGE2TEXT = 'image2text'
    RERANK = 'rerank'
    TTS = 'tts'


class ChatStyle(StrEnum):
    CREATIVE = 'Creative'
    PRECISE = 'Precise'
    EVENLY = 'Evenly'
    CUSTOM = 'Custom'


class TaskStatus(StrEnum):
    UNSTART = "0"
    RUNNING = "1"
    CANCEL = "2"
    DONE = "3"
    FAIL = "4"


class ParserType(StrEnum):
    PRESENTATION = "presentation"
    LAWS = "laws"
    MANUAL = "manual"
    PAPER = "paper"
    BOOK = "book"
    QA = "qa"
    TABLE = "table"
    NAIVE = "naive"
    ONE = "one"
    EMAIL = "email"
    KG = "knowledge_graph"
    TAG = "tag"


class FileSource(StrEnum):
    LOCAL = ""
    KNOWLEDGEBASE = "knowledgebase"
    S3 = "s3"


class CanvasType(StrEnum):
    ChatBot = "chatbot"
    DocBot = "docbot"


# Constants
KNOWLEDGEBASE_FOLDER_NAME = ".knowledgebase"


def get_project_base_directory():
    """Get project base directory (replaces api.utils.file_utils.get_project_base_directory)"""
    return Path(__file__).parent.parent.parent.absolute()

def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


def rmSpace(txt):
    txt = re.sub(r"([^a-z0-9.,\)>]) +([^ ])", r"\1\2", txt, flags=re.IGNORECASE)
    return re.sub(r"([^ ]) +([^a-z0-9.,\(<])", r"\1\2", txt, flags=re.IGNORECASE)


def findMaxDt(fnm):
    m = "1970-01-01 00:00:00"
    try:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip("\n")
                if line == 'nan':
                    continue
                if line > m:
                    m = line
    except Exception:
        pass
    return m

  
def findMaxTm(fnm):
    m = 0
    try:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip("\n")
                if line == 'nan':
                    continue
                if int(line) > m:
                    m = int(line)
    except Exception:
        pass
    return m


tiktoken_cache_dir = get_project_base_directory()
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])
