import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from typing import Optional
from ..qlinear import QuantLinear
from ..gptq import *
from ..quant import *
from QQQ.utils import find_layers
from transformers.utils import logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def gptq_llama_func(model, dataloader, dev, args, force_to_cpu=False):
    print("Starting GPTQ quantization ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
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

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quantizers = {}
    for i, layer in enumerate(layers):
        if layer.input_layernorm.weight.device == torch.device("cpu"):
            layer = layer.to(dev)
        cur_device = layer.input_layernorm.weight.device
        inps = inps.to(cur_device)
        outs = outs.to(cur_device)
        attention_mask = attention_mask.to(cur_device) if attention_mask is not None else None
        position_ids = position_ids.to(cur_device)

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
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
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
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
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


class QuantizedLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: dict,
        layer_idx: Optional[int] = None,
    ):
        super(LlamaAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.q_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

class QuantizedLlamaFlashAttention2(LlamaFlashAttention2, QuantizedLlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        QuantizedLlamaAttention.__init__(self, *args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


class QuantizedLlamaSdpaAttention(LlamaSdpaAttention, QuantizedLlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def __init__(self, *args, **kwargs):
        QuantizedLlamaAttention.__init__(self, *args, **kwargs)


QUANT_LLAMA_ATTENTION_CLASSES = {
    "eager": QuantizedLlamaAttention,
    "flash_attention_2": QuantizedLlamaFlashAttention2,
    "sdpa": QuantizedLlamaSdpaAttention,
}

class QuantizedLlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, quant_config: dict):
        super(LlamaMLP, self).__init__()
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
        self.act_fn = ACT2FN[config.hidden_act]


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self, config: LlamaConfig, quant_config: dict, layer_idx: int
    ):
        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QUANT_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, quant_config=quant_config, layer_idx=layer_idx)
        self.mlp = QuantizedLlamaMLP(config, quant_config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class QuantizedLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, quant_config: dict):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # no quant on embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                QuantizedLlamaDecoderLayer(config, quant_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_position_embeddings`.
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()


class QuantizedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, quant_config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = QuantizedLlamaModel(config, quant_config)
        self.vocab_size = config.vocab_size
        # no quant on lm_head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

