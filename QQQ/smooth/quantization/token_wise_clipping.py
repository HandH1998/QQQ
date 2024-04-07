from torch.nn import MSELoss
import logging
from smooth.quantization.fake_quant import QuantizeBase

logger = logging.getLogger("QQQ")


def set_ratio(model, ratio):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if "act" in name:
                module.observer.set_percentile(ratio)
                # module.observer.cnt = 0
                module.disable_fake_quant()
                module.enable_observer()
            if "weight" in name:
                module.disable_fake_quant()


def enable_quantization(model):
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if "act" in name:
                submodule.disable_observer()
                submodule.enable_fake_quant()
            if "weight" in name:
                submodule.enable_fake_quant()


def calibrate(model, fp_input, fp_output=False):
    loss = 0
    for i, batch in enumerate(fp_input):
        if fp_output:
            loss += model(**batch, labels=fp_input[i]["input_ids"]).loss
        else:
            model(**batch)
    return loss


def find_ratio(model, fp_input, fp_output, param):
    p, loss = 0, None
    iters = param["iters"]
    step = param["step"]
    for i in range(iters):
        set_ratio(model, 1.0 - step * i)
        calibrate(model, fp_input)
        enable_quantization(model)
        cur_loss = calibrate(model, fp_input, True)
        logger.info("the ratio is {}, the loss is {}".format(1.0 - step * i, cur_loss))
        if loss is None or loss > cur_loss:
            loss = cur_loss
            p = i
    ratio = 1.0 - step * p
    logger.info("the best percentile is {}".format(ratio))
    set_ratio(model, ratio)
    calibrate(model, fp_input)


loss_fct = MSELoss()


a_bit_iters = {
    8: 0.05,
    6: 0.1,
}


def cac_step_iters(a_bit, bs):
    step = 0.005
    step = float(format(step, ".2g"))
    iters = int(a_bit_iters[a_bit] / step)
    print("the step is {}, the iter is {}".format(step, iters))
    return step, iters


def token_wise_clipping(model, fp_input, fp_output, config_quant, batch_size):
    # config_quant = config.quant

    logger.info("*** Evaluate Token Percentile ***")
    step, iters = cac_step_iters(config_quant.a_qconfig.bit, batch_size)

    if hasattr(config_quant.a_qconfig, "token_quantile"):
        set_ratio(model, config_quant.a_qconfig.token_quantile)
        calibrate(model, fp_input)
        logger.info(
            "the best percentile is {}".format(config_quant.a_qconfig.token_quantile)
        )
    else:
        step, iters = cac_step_iters(config_quant.a_qconfig.bit, batch_size)
        find_ratio(
            model,
            fp_input,
            fp_output,
            {
                "iters": getattr(config_quant, "iters", iters),
                "step": getattr(config_quant, "step", step),
            },
        )
