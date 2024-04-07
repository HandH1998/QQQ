from .quant_llama import QuantizedLlamaForCausalLM
from QQQ.smooth.quantization.observer import ObserverBase
from QQQ.utils import prepare_for_inference


def quantize_model(fp_model, config_quant, args):
    # config_quant = config.quant
    config_quant.is_remove_padding = config_quant.get("is_remove_padding", True)
    config_quant.migrate = config_quant.get("migrate", False)
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
