# Adapted from https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_marlin.py
# Modified by HandH1998
# Copyright (C) 2024 HandH1998
# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from QQQ._CUDA import qqq_gemm


logger = getLogger(__name__)


def mul(A, B, C, D, s1, s2, s3, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """INT8xINT4 multiply based on Marlin kernel; can be used within `torch.compile`.
    @A: `torch.int8` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in the specified format; see `Layer.pack()`
    @C: `torch.int` reduce buffer of shape `(max_par * 64, n)` in standard row-major layout
    @D: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s1: `torch.float` activation per-token quantization scales of shape `(m, 1)`
    @s2: `torch.float` weight per-channel quantization scales of shape `(1, n)`
    @s3: `torch.half` weight per-group quantization scales of shape `(m / groupsize, n)`, it should be empty when group_size != -1
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    qqq_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms, max_par)


class QuantLinear(nn.Module):
    QUANT_TYPE = "marlin"

    def __init__(
        self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs
    ):
        super().__init__()

        if torch.version.hip:
            raise ValueError(
                "Can not use Marlin int4*fp16 kernel with AMD ROCm version of PyTorch as the kernel is not compatible. Please do not use `use_marlin=True` when using ROCm devices."
            )
        if not torch.cuda.get_device_capability()[0] >= 8:
            raise ValueError(
                f'Can not use Marlin int4*fp16 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel. Please do not use `use_marlin=True`, or please upgrade your GPU ("The more you buy, the more you save." - Taiwanese proverb).'
            )

        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        if group_size not in [-1, 128] and group_size != infeatures:
            raise ValueError("Only group_size -1 and 128 are supported.")
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")
        if trainable:
            raise NotImplementedError("Marlin does not support train.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = group_size if group_size != -1 else infeatures
        self.bits = bits
        self.tile = 16
        if self.group_size != self.infeatures:
            self.maxq = 2**self.bits - 1
        else:
            self.maxq = 2**(self.bits - 1) - 1
        self.max_par = 16
        self.register_buffer(
            "B",
            torch.empty(
                (self.infeatures // 16, self.outfeatures * 16 // 8), dtype=torch.int
            ),
        )
        self.register_buffer(
            "s_channel",
            torch.empty(
                (1, self.outfeatures),
                dtype=torch.float,
            ),
        )
        if self.group_size != self.infeatures:
            self.register_buffer(
                "s_group",
                torch.empty(
                    (self.infeatures // self.group_size, self.outfeatures), dtype=torch.half
                ),
            )
        else:
            self.register_buffer(
                "s_group",
                torch.tensor(
                    [], dtype=torch.half
                ),
            )
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer(
            "workspace",
            torch.zeros(self.outfeatures // 128 * 16, dtype=torch.int),
            persistent=False,
        )
        self.register_buffer(
            "reduce_buffer",
            torch.zeros((self.max_par * 16 * 4, self.outfeatures), dtype=torch.int),
            persistent=False,
        )
        self.wf = torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0)
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.half))
        else:
            self.bias = None
        self._perm, self._scale_perm, self._scale_perm_single = self._get_perms()

    def _apply(self, fn):
        super()._apply(fn)
        self.s_group = self.s_group.to(torch.half)
        self.s_channel = self.s_channel.to(torch.float)
        return self

    def _get_perms(self):
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])

        perm = np.array(perm)
        if self.group_size == self.infeatures:
            interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
        perm = torch.from_numpy(perm)
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        return perm, scale_perm, scale_perm_single


    def post_init(self):
        pass

    def pack(self, linear, scales, s_extra=None):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        @s_extra: corresponding quantization scales of shape `(1, outfeatures)`
        """
        if self.group_size != self.infeatures:
            assert s_extra is not None, "s_extra is needed"
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        s = scales.t()
        w = linear.weight.data.t()
        if self.group_size != self.infeatures:
            w = w.reshape((-1, self.group_size, self.outfeatures))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.group_size, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        if self.group_size != self.infeatures:
            w += (self.maxq + 1) // 2
            w = torch.clamp(w, 0, self.maxq)
        else:
            w = torch.clamp(w, -self.maxq, self.maxq)
        if self.group_size != self.infeatures:
            s_extra = s_extra.reshape(1, -1).to(dtype=torch.float)
            s = (
                s.reshape(-1, self.outfeatures) / s_extra
            ).to(dtype=torch.half)

            w = w.reshape((self.group_size, -1, self.outfeatures))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.infeatures, self.outfeatures)).contiguous()
            s = s.reshape((-1, len(self._scale_perm)))[:, self._scale_perm]
            s_extra = s_extra.reshape((-1, len(self._scale_perm_single)))[
                :, self._scale_perm_single
            ]
            s_extra = s_extra.reshape((-1, self.outfeatures)).contiguous()
        else:
            # NOTE(zhangying): div 2 ** (8 - self.bits)) to deal with right_shift in unpacking
            s = (s / (2 ** (8 - self.bits))).reshape((-1, len(self._scale_perm_single)))[
                :, self._scale_perm_single
            ].to(dtype=torch.float)
        s = s.reshape((-1, self.outfeatures)).contiguous()
        w = w.reshape(
            (
                self.infeatures // self.tile,
                self.tile,
                self.outfeatures // self.tile,
                self.tile,
            )
        )
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.infeatures // self.tile, self.outfeatures * self.tile))
        res = w
        res = res.reshape((-1, self._perm.numel()))[:, self._perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        if self.group_size != self.infeatures:
            for i in range(8):
                q |= res[:, i::8] << 4 * i
        else:
            for i in range(8):
                q |= (res[:, i::8] & 0xF) << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        if self.group_size != self.infeatures:
            self.s_group[:, :] = s.to(self.s_group.device)
            self.s_channel[:, :] = s_extra.to(self.s_channel.device)
        else:
            self.s_group = torch.tensor([], dtype=torch.half, device=self.s_channel.device)
            self.s_channel[:, :] = s.to(self.s_channel.device)
        if linear.bias is not None:
            if self.bias is not None:
                self.bias[:] = linear.bias.data.to(self.bias.device)
            else:
                self.bias = linear.bias.clone()

    # activation int8 quantization
    def dynamic_quant(self, x: torch.Tensor):
        quant_scale = x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float)
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        return x, quant_scale
    
    def forward(self, A):
        out_shape = A.shape[:-1] + (self.outfeatures,)
        A = A.reshape(-1, A.shape[-1]).half()
        quant_A, s1 = self.dynamic_quant(A)
        D = torch.empty(A.shape[0], self.outfeatures, dtype=A.dtype, device=A.device)
        mul(
            quant_A,
            self.B,
            self.reduce_buffer,
            D,
            s1,
            self.s_channel,
            self.s_group,
            self.workspace,
            max_par=self.max_par
        )
        D = D.reshape(out_shape)
        D = D + self.bias if self.bias is not None else D
        return D

__all__ = ["QuantLinear"]
