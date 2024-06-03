import logging
from .fake_quant import QuantizeBase
from .observer import ObserverBase

logger = logging.getLogger("QQQ")

def enable_calibration_quantization(
    model, quantizer_type="fake_quant"
):
    logger.info("Enable observer and Enable quantize for {}".format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug("The except_quantizer is {}".format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug("Enable observer and Enable quant: {}".format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    logger.info("Disable observer and disable quantize.")
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.debug("Disable observer and disable quant: {}".format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()


def set_observer_name(model):
    logger.info("set name for obsever")
    for name, submodule in model.named_modules():
        if isinstance(submodule, ObserverBase):
            submodule.set_name(name)
