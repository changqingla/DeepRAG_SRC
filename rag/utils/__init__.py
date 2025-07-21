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


def get_project_base_directory():
    """Get project base directory"""
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

def get_home_cache_dir():
    dir = os.path.join(os.path.expanduser('~'), ".deeprag")
    try:
        os.mkdir(dir)
    except OSError:
        pass
    return dir

def log_exception(e, *args):
    logging.exception(e)
    for a in args:
        if hasattr(a, "text"):
            logging.error(a.text)
            raise Exception(a.text)
        else:
            logging.error(str(a))
    raise e

def conf_realpath(conf_name):
    conf_path = f"conf/{conf_name}"
    return os.path.join(file_utils.get_project_base_directory(), conf_path)


# def read_config(conf_name=SERVICE_CONF):
#     local_config = {}
#     local_path = conf_realpath(f'local.{conf_name}')

#     # load local config file
#     if os.path.exists(local_path):
#         local_config = file_utils.load_yaml_conf(local_path)
#         if not isinstance(local_config, dict):
#             raise ValueError(f'Invalid config file: "{local_path}".')

#     global_config_path = conf_realpath(conf_name)
#     global_config = file_utils.load_yaml_conf(global_config_path)

#     if not isinstance(global_config, dict):
#         raise ValueError(f'Invalid config file: "{global_config_path}".')

#     global_config.update(local_config)
#     return global_config

# CONFIGS = read_config()

# def get_base_config(key, default=None):
#     if key is None:
#         return None
#     if default is None:
#         default = os.environ.get(key.upper())
#     return CONFIGS.get(key, default)

# def decrypt_database_config(
#         database=None, passwd_key="password", name="database"):
#     if not database:
#         database = get_base_config(name, {})

#     database[passwd_key] = decrypt_database_password(database[passwd_key])
#     return database

# def decrypt_database_password(password):
#     encrypt_password = get_base_config("encrypt_password", False)
#     encrypt_module = get_base_config("encrypt_module", False)
#     private_key = get_base_config("private_key", None)

#     if not password or not encrypt_password:
#         return password

#     if not private_key:
#         raise ValueError("No private key")

#     module_fun = encrypt_module.split("#")
#     pwdecrypt_fun = getattr(
#         importlib.import_module(
#             module_fun[0]),
#         module_fun[1])

#     return pwdecrypt_fun(private_key, password)

def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname