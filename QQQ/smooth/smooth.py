import torch
import logging
import os
import time
from torch.nn.utils.rnn import pad_sequence

from .quantization.quantized_module import QuantizedModule
from .quantization.state import (
    enable_calibration_quantization,
    disable_all,
)
from QQQ.utils import get_model_architecture, get_loaders, str2torch_device

logger = logging.getLogger("QQQ")


@torch.no_grad()
def calibrate_batch(model, fp_input):
    logger.info("*** Calibrate ***")
    for batch in fp_input:
        model(**batch)


def create_batches(tokenizer, dataloader, batch_size, device):
    logger.info("**prepare fp input and output**")
    fp_input, fp_output = [], []
    input_ids_list = [inp[0].squeeze(0) for inp in dataloader]
    batches = [
        input_ids_list[i : i + batch_size]
        for i in range(0, len(input_ids_list), batch_size)
    ]

    for batch in batches:
        tmp = {}
        padded_input_ids = pad_sequence(
            batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()
        tmp["input_ids"] = padded_input_ids.to(device)
        tmp["attention_mask"] = attention_mask.to(device)
        fp_input.append(tmp)

    return fp_input, fp_output


@torch.no_grad()
def smooth(model, tokenizer, smooth_config, args):
    logger.info("the quantization config is {}".format(smooth_config))
    logger.info("begin building calibration data!")
    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        seqlen=args.max_length,
        custom_data_path=args.custom_dataset,
    )
    device = str2torch_device(args.device)
    fp_input, fp_output = create_batches(
        tokenizer, dataloader, smooth_config.batch_size, device
    )

    logger.info("begin smooth!")
    st = time.time()
    enable_calibration_quantization(model)
    model_type = get_model_architecture(model.config)
    if model_type == "llama":
        from .migration import migration_llama as migration
    elif model_type == "qwen2":
        from .migration import migration_qwen2 as migration
    else:
        raise NotImplementedError
    migration.set_search_class(smooth_config.smooth_method)

    for name, module in model.named_modules():
        if isinstance(module, QuantizedModule):
            module.set_cac_migrate(True)
    calibrate_batch(model, [fp_input[0]])
    for name, module in model.named_modules():
        if isinstance(module, QuantizedModule):
            module.set_cac_migrate(False)

    # save smooth scale
    torch.save(
        migration.scale_list,
        os.path.join(args.save_path, "scale_list.pth"),
    )

    # activation clip
    if smooth_config.a_qconfig.observer == "QuantileObserver":
        from .quantization.token_wise_clipping import token_wise_clipping

        disable_all(model)
        token_wise_clipping(model, fp_input, fp_output, smooth_config, args.batch_size)

    ed = time.time()
    logger.info("cost {:.4f} time".format(ed - st))
    return migration.scale_list
