from QQQ.smooth.models import QuantizedQwen2ForCausalLM, QuantizedLlamaForCausalLM
from QQQ.smooth.quantization.observer import ObserverBase
from QQQ.utils import prepare_for_inference


def quantize_model(fp_model, config_quant, args):
    fp_model.eval()
    model = eval("Quantized" + str(fp_model.__class__.__name__))(
        fp_model,
        config_quant.w_qconfig,
        config_quant.a_qconfig,
        qinput=False,
        is_remove_padding=config_quant.is_remove_padding,
    )
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase) and "act" in name:
            module.set_name(name)
    model = prepare_for_inference(model, args.device, args.dtype)
    return model
