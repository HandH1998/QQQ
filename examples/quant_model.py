import argparse
import logging
import os
import torch
from QQQ.smooth import smooth, export_smoothed_model, quantize_model
from QQQ.gptq import apply_gptq
from QQQ.utils import (
    setup_seed,
    parse_config,
    build_model_and_tokenizer,
    prepare_for_inference,
    free_memory,
)
logger = logging.getLogger("QQQ")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--smooth_method", default="os+", choices=["os+", "awq"])
    parser.add_argument("--quant_config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    # set seed
    setup_seed(args.seed)

    # pase config
    if args.quant_config:
        q_config = parse_config(args.quant_config)
    else:
        q_config = None

    # process save_path
    if args.save_path:
        sub_dir_name = args.model_path.split("/")[-1]
        args.save_path = os.path.join(args.save_path, sub_dir_name)
        os.makedirs(args.save_path, exist_ok=True)

    # tokenizer path
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # load model
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype
    )

    # smooth model
    model = quantize_model(model, q_config, args)
    scale_list = smooth(model, tokenizer, q_config, args)
    del model
    del tokenizer
    free_memory()

    # load model and apply smooth scales
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype
    )
    model = export_smoothed_model(model, scale_list)

    # apply gptq
    model = prepare_for_inference(model, args.device, args.dtype)
    model = apply_gptq(model, q_config, args)

    # quant_config
    model.config.quantization_config = {
        "group_size": q_config["gptq"]["groupsize"],
        "quant_method": "qqq",
        "wbits": q_config["gptq"]["wbits"]
    }

    # save quantized model
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    logger.info("Quant Finished! The quantized model is saved at {}.".format(args.save_path))


if __name__ == "__main__":
    main()
