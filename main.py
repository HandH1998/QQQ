import argparse
import logging
import os
import torch
import re
import collections
import json
from tqdm import tqdm
import lm_eval
from lm_eval import tasks, simple_evaluate
from lm_eval.models.huggingface import HFLM

from QQQ.smooth import smooth, export_smoothed_model, quantize_model
from QQQ.gptq import apply_gptq
from QQQ.utils import (
    setup_seed,
    parse_config,
    build_model_and_tokenizer,
    get_loaders,
    pattern_match,
    update_results,
    MultiChoice,
    get_max_length,
    prepare_for_inference,
    free_memory
)

logger = logging.getLogger("QQQ")


def eval_model(model, tokenizer, args):
    model = prepare_for_inference(model, args.device, args.dtype)
    max_length = get_max_length(model)
    results = {}
    # eval ppl
    for task in ["wikitext2"]:
        _, testloader = get_loaders(
            task,
            seed=0,
            model=args.tokenizer_path,
            seqlen=max_length,
        )
        if "c4" in task:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // max_length
   
        nlls = []
        for i in tqdm(range(nsamples)):
            batched_inps = testenc[
                :, (i * max_length) : ((i + 1) * max_length)
            ].to("cuda:0")
            batched_labels = testenc[
                :, (i * max_length) : ((i + 1) * max_length)
            ].to("cuda:0")
            loss = model(batched_inps, labels=batched_labels).loss
            neg_log_likelihood = loss.float() * max_length
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_length))

        result = collections.defaultdict(dict)
        versions = collections.defaultdict(dict)
        n_shot = collections.defaultdict(dict)
        result[task]["ppl"] = ppl.item()
        versions[task] = 0
        n_shot[task] = 0
        t_results = {
            "results": dict(result),
            "versions": dict(versions),
            "n-shot": dict(n_shot),
        }
        print(t_results)
        update_results(results, t_results)
    # eval other datasets
    if args.tasks != "":
        task_names = pattern_match(args.tasks.split(","), tasks.TaskManager().all_tasks)
        lm = HFLM(pretrained=model, backend='causal', device='cuda', batch_size=args.batch_size, tokenizer=tokenizer)
        t_results = simple_evaluate(
            lm,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
        )
        update_results(results, t_results)

    dumped = json.dumps(results, indent=2)
    with open(os.path.join(args.save_path, "eval_result.json"), "w") as f:
        f.write(dumped)
    print(lm_eval.utils.make_table(results))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--smooth_method", default="os+", choices=["os+", "awq"])
    parser.add_argument("--quant_config", type=str, default=None)
    parser.add_argument("--eval_model", action="store_true", default=True)
    parser.add_argument(
        "--tasks", default="", choices=MultiChoice(tasks.TaskManager().all_tasks)
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
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
        sep = "[/.]+"
        sub_dir_name = "-".join(re.split(sep, args.quant_config)[-3:-1])
        args.save_path = os.path.join(args.save_path, sub_dir_name)
        os.makedirs(args.save_path, exist_ok=True)

    # tokenizer path
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # load model
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype, args.device
    )

    # smooth model
    model = quantize_model(model, q_config, args)
    scale_list = smooth(model, tokenizer, q_config, args)
    del model
    del tokenizer
    free_memory()
    
    # load model and apply smooth scales
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype, args.device
    )
    model = export_smoothed_model(model, scale_list)

    # apply gptq
    model = apply_gptq(model, q_config, args)

    # save quantized model
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    # eval model
    if args.eval_model:
        eval_model(model, tokenizer, args)


if __name__ == "__main__":
    main()
