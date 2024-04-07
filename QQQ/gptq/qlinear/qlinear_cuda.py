import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers
from .int8gemm import Int8GEMM


logger = getLogger(__name__)


class QuantLinear(nn.Module):
    QUANT_TYPE = "cuda"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** (self.bits - 1) - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [i // self.group_size for i in range(infeatures)], dtype=torch.int32
            ),
        )
        if self.group_size != self.infeatures:
            self.register_buffer(
                "int8_scales",
                torch.zeros(
                    (1, outfeatures),
                    dtype=weight_dtype,
                ),
            )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

        # is performed by unpacking the weights and using torch.matmul
        self.wf = torch.tensor(
            list(range(0, 32, self.bits)), dtype=torch.int32
        ).unsqueeze(0)
        self.trainable = trainable
        GEMM = Int8GEMM()
        self.i8cugemm = GEMM.get_i8cugemm()

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx=None, int8_scales=None):
        if self.group_size != self.infeatures:
            assert int8_scales is not None, "int8_scales is needed"
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        step = 32 // self.bits
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scale_zeros = zeros * scales
        intweight = (
            torch.round(
                (W.reshape(-1, self.group_size) + scale_zeros.reshape(-1, 1))
                / scales.reshape(-1, 1)
            )
            .reshape(self.outfeatures, self.infeatures)
            .to(torch.int)
        )
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        for i in range(step):
            qweight |= (intweight[i::step, :] & 0xF) << self.bits * i

        self.qweight = torch.from_numpy(qweight.astype(np.int32))

        scales = scales.t().contiguous()
        # NOTE(zhangying): div 2 ** (8 - self.bits)) to deal with right_shift in unpacking
        if self.group_size != self.infeatures:
            self.scales = (
                scales / (int8_scales.reshape(1, -1) * (2 ** (8 - self.bits)))
            ).to(dtype=linear.weight.dtype)
            self.int8_scales = int8_scales.reshape(1, -1).to(dtype=linear.weight.dtype)
        else:
            self.scales = (
                (scales / (2 ** (8 - self.bits))).clone().to(dtype=linear.weight.dtype)
            )
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        # zeros -= 1
        zeros = zeros.t().contiguous()
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
        )
        for i in range(step):
            qzeros |= (zeros[:, i::step] & 0xF) << self.bits * i
        self.qzeros = torch.from_numpy(qzeros.astype(np.int32))

    # int8 quant activation
    def dynamic_quant(self, x: torch.Tensor):
        quant_scale = x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        return x, quant_scale

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        # unpack zero
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int8)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)
        zeros = zeros[:, :, :] << (8 - self.bits)
        # zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        # unpack weight
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight[:, :, :] << (8 - self.bits)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if self.group_size == self.infeatures:
            weights = weight - zeros[self.g_idx.long()]
        else:
            weights = self.scales[self.g_idx.long()] * (
                weight - zeros[self.g_idx.long()]
            )
            weights = weights.round().clamp(-128, 127).to(torch.int8)

        # quant activation
        quant_x, x_scales = self.dynamic_quant(x)

        # int8 GEMM
        out = torch.empty(
            x.shape[0],
            self.outfeatures,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.i8cugemm.linear_a8_w8_o32_(quant_x, weights.t().contiguous(), out)

        # dequant
        # NOTE(zhangying): scale need to cast to float32, as float16 will overflow when multiplying.
        weight_scales = (
            self.scales.to(torch.float32)
            if self.group_size == self.infeatures
            else self.int8_scales.to(torch.float32)
        )

        dequant_scales = x_scales * weight_scales
        
        out = (out * dequant_scales).to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["QuantLinear"]
