import torch
import torch.nn as nn
import copy
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RMSNorm,
    Gemma3MLP,
    Gemma3RotaryEmbedding,
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3Model,
    Gemma3TextScaledWordEmbedding,
    Gemma3ForCausalLM,
)
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.activations import ACT2FN
from ..qlinear import QuantLinear
from ..gptq import *
from ..quant import *
from QQQ.utils import find_layers
from transformers.utils import logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def gptq_gemma3_func(model, dataloader, dev, args, force_to_cpu=False):
    print("Starting GPTQ quantization ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    inps = []
    attention_mask = []
    position_ids = []
    cache_position = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            cache_position.append(kwargs["cache_position"])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if force_to_cpu:
        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()

    outs = [inp.clone() for inp in inps]

    quantizers = {}
    for i, layer in enumerate(layers):
        if layer.input_layernorm.weight.device == torch.device("cpu"):
            layer = layer.to(dev)
        cur_device = layer.input_layernorm.weight.device
        inps = [inp.to(cur_device) for inp in inps]
        outs = [out.to(cur_device) for out in outs]
        attention_mask = [
            att_mask.to(cur_device) if att_mask is not None else None
            for att_mask in attention_mask
        ]
        position_ids = [pos_ids.to(cur_device) for pos_ids in position_ids]
        cache_position = [
            cache_pos.to(cur_device) if cache_pos is not None else None
            for cache_pos in cache_position
        ]

        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits,
                    perchannel=True,
                    sym=args.sym,
                    mse=args.mse,
                    groupsize=args.groupsize,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                    cache_position=cache_position[j],
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                scale, zero, g_idx, scale_extra = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = (
                    scale,
                    zero,
                    g_idx,
                    scale_extra,
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask[j],
                position_ids=position_ids[j],
                cache_position=cache_position[j],
            )[0]

        if force_to_cpu:
            layers[i] = layer.cpu()
            del layer
        else:
            layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


class QuantizedGemma3Attention(Gemma3Attention):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int, quant_config: dict):
        super().__init__()
        self.is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]

        self.q_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = QuantLinear(
            wbits,
            group_size,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = QuantLinear(
            wbits,
            group_size,
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)


class QuantizedGemma3MLP(Gemma3MLP):
    def __init__(self, config: Gemma3TextConfig, quant_config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]

        self.gate_proj = QuantLinear(
            wbits, group_size, self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = QuantLinear(
            wbits, group_size, self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = QuantLinear(
            wbits, group_size, self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_activation]


class QuantizedGemma3DecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int, quant_config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = QuantizedGemma3Attention(
            config=config, layer_idx=layer_idx, quant_config=quant_config
        )
        self.mlp = QuantizedGemma3MLP(config, quant_config=quant_config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window


class QuantizedGemma3TextModel(Gemma3Model):
    def __init__(self, config: Gemma3TextConfig, quant_config: dict):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                QuantizedGemma3DecoderLayer(
                    config, layer_idx, quant_config=quant_config
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()


class QuantizedGemma3ForCausalLM(Gemma3ForCausalLM):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = QuantizedGemma3TextModel(config)
        self.vocab_size = config.vocab_size
        # no quant on lm_head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
