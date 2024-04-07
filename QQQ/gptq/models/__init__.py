from .llama import gptq_llama_func

_GPTQ_MODEL_FUNC = {
    "llama": gptq_llama_func
}

def get_gptq_model_func(model_type):
    if model_type in _GPTQ_MODEL_FUNC:
        return _GPTQ_MODEL_FUNC[model_type]
    else:
        raise NotImplementedError