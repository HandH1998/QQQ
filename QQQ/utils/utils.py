import fnmatch
import yaml
import os
import numpy as np
import torch
from easydict import EasyDict
import random
import json
import gc


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}


def str2torch_dtype(dtype):
    if dtype in DTYPE_MAP:
        torch_dtype = DTYPE_MAP[dtype]
    else:
        raise ValueError("Not supported dtype: {}!".format(dtype))
    return torch_dtype


def str2torch_device(device):
    if device:
        if device not in ["cuda", "cpu"]:
            device = int(device)
        device = torch.device(device)
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return device


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while "root" in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config["root"])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_json(config, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)


def parse_quant_config(config_path):
  data = {}
  with open(config_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  return data

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

