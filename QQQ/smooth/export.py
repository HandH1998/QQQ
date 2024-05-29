import torch
import argparse
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from QQQ.utils import build_model_and_tokenizer, get_model_architecture


def export_smoothed_llama(model, scale_list):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm
            q, k, v, o = (
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
                module.self_attn.o_proj,
            )
            # attn ln
            attn_ln.weight.data /= scale_list[cnt].to(attn_ln.weight.data.device)
            # qkv
            q.weight.data *= scale_list[cnt].to(q.weight.data.device)
            k.weight.data *= scale_list[cnt].to(k.weight.data.device)
            v.weight.data *= scale_list[cnt].to(v.weight.data.device)
            cnt += 1

            # no smoothing o_proj for models using group attention
            if module.self_attn.num_key_value_heads == module.self_attn.num_heads:
                o.weight.data *= scale_list[cnt].to(o.weight.data.device)
                v.weight.data /= scale_list[cnt].reshape(-1, 1).to(v.weight.data.device)
            cnt += 1

            ffn_ln = module.post_attention_layernorm
            gate = module.mlp.gate_proj
            up = module.mlp.up_proj
            down = module.mlp.down_proj
            # ffn ln
            ffn_ln.weight.data /= scale_list[cnt].to(ffn_ln.weight.data.device)
            # gate up
            gate.weight.data *= scale_list[cnt].to(gate.weight.data.device)
            up.weight.data *= scale_list[cnt].to(up.weight.data.device)
            cnt += 1

            down.weight.data *= scale_list[cnt].to(down.weight.data.device)
            up.weight.data /= scale_list[cnt].reshape(-1, 1).to(up.weight.data.device)
            cnt += 1

    return model


def export_smoothed_model(model, scale_list):
    model_type = get_model_architecture(model.config)
    if model_type == "llama":
        model = export_smoothed_llama(model, scale_list)
    else:
        raise NotImplementedError
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--scale_list", required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype, args.device
    )
    scale_list = torch.load(args.scale_list)
    model = export_smoothed_model(model, scale_list)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
