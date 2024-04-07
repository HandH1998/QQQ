import torch
import torch.nn as nn
import functools
import gc
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from .utils import str2torch_dtype, str2torch_device

_MODEL_TYPE = {
    "LlamaForCausalLM": "llama",
    "LLaMAForCausalLM": "llama",
}


def build_model_and_tokenizer(
    model_path, tokenizer_path, dtype: str, device: str, trust_remote_code: bool = True
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"torch_dtype": str2torch_dtype(dtype), "device_map": "auto", "attn_implementation": "eager"}
    # kwargs = {"torch_dtype": str2torch_dtype(dtype), "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, **kwargs
    )
    model = model.to(str2torch_device(device))
    return model, tokenizer


def get_model_architecture(config):
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_TYPE:
            return _MODEL_TYPE[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_TYPE.keys())}"
    )

def get_max_length(model):
    try:
        return model.config.n_ctx
    except AttributeError:
        # gptneoconfig doesn't have n_ctx apparently
        return model.config.max_position_embeddings
    
def prepare_for_inference(model, device, dtype):
    model.to(str2torch_device(device))
    model.to(str2torch_dtype(dtype))
    model.eval()
    return model

def free_memory():
    # del model
    gc.collect()
    torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

import functools


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)
