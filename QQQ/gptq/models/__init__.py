from .llama import gptq_llama_func, QuantizedLlamaForCausalLM

_GPTQ_MODEL_FUNC = {
    "llama": gptq_llama_func
}

_QUANTIZED_MODEL_CLASS = {
    "llama": QuantizedLlamaForCausalLM
}

def get_gptq_model_func(model_type):
    if model_type in _GPTQ_MODEL_FUNC:
        return _GPTQ_MODEL_FUNC[model_type]
    else:
        raise NotImplementedError
    
def get_quantized_model_class(model_type):
    if model_type in _QUANTIZED_MODEL_CLASS:
        return _QUANTIZED_MODEL_CLASS[model_type]
    else:
        raise NotImplementedError