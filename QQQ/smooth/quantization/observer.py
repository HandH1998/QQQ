import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import os
import logging
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
from .quant_utils import (
    fake_quantize_per_tensor_affine,
    fake_quantize_per_channel_affine,
)

logger = logging.getLogger("QQQ")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _transform_to_ch_axis(x, ch_axis):
    if ch_axis == -1:
        return x
    else:
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[ch_axis] = 0
        new_axis_list[0] = ch_axis
        x_channel = x.permute(new_axis_list)
        y = torch.flatten(x_channel, start_dim=1)
        return y


class ObserverBase(nn.Module):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(ObserverBase, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.register_buffer("eps", torch.tensor([torch.finfo(torch.float).eps]))
        if self.symmetric:
            self.quant_min = -(2 ** (self.bit - 1)) + 1
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2**self.bit - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def set_name(self, name):
        self.name = name

    def set_batch(self, batch):
        self.batch = batch

    def set_percentile(self, percentile):
        self.percentile = percentile

    def quantile_range(self, x, percentile):
        upper = torch.quantile(x.to(torch.float32).abs(), percentile)
        return -upper, upper

    def cac_thres(self, token_min, token_max):
        _, upper = self.quantile_range(token_max, self.percentile)
        lower, _ = self.quantile_range(token_min, self.percentile)
        indice_upper = torch.nonzero(token_max <= upper, as_tuple=True)[0]
        indice_lower = torch.nonzero(token_min >= lower, as_tuple=True)[0]
        return indice_lower, indice_upper

    def prune_token(self, value):  # try batch first
        if "attention_probs" in self.name:
            return value
        token_max, _ = value.max(1)
        token_min, _ = value.min(1)
        indice_lower, indice_upper = self.cac_thres(token_min, token_max)
        upper = token_max[indice_upper].max()
        lower = token_min[indice_lower].min()
        value = torch.clip(value, max=upper, min=lower)
        return value

    def remove_padding(self, x, observation_mask, seq_pos):
        # assert the first dim is batch
        pos = list(range(len(x.shape)))
        shape = x.shape
        pos.remove(seq_pos)
        if len(pos) == 3:
            x = x.permute(pos[0], seq_pos, pos[1], pos[2]).reshape(
                shape[pos[0]], shape[seq_pos], -1
            )
        if len(pos) == 2:
            x = x.permute(pos[0], seq_pos, pos[1])
        return x[observation_mask == 1]

    def reshape_batch_embedding(self, x, seq_pos):
        # assert the first dim is batch
        pos = list(range(len(x.shape)))
        shape = x.shape
        pos.remove(seq_pos)
        if len(pos) == 3:
            x = x.permute(pos[0], seq_pos, pos[1], pos[2]).reshape(
                shape[pos[0]], shape[seq_pos], -1
            )
        if len(pos) == 2:
            x = x.permute(pos[0], seq_pos, pos[1])
        return x.reshape(shape[pos[0]] * shape[seq_pos], -1)

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=min_val_neg.dtype, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    """
    Calculate minmax of whole calibration dataset.
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = min_val_cur
        self.max_val = max_val_cur
        return min_val_cur, max_val_cur


class QuantileObserver(ObserverBase):
    """
    Calculate minmax of whole calibration dataset.
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(QuantileObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        self.percentile = 1.0

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            y = self.prune_token(y)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = min_val_cur
        self.max_val = max_val_cur
        return min_val_cur, max_val_cur


class LSQPlusObserver(ObserverBase):
    """
    LSQ+ observer. This only suits for weight Observer
    """

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(LSQPlusObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        assert self.symmetric is True
        self.mean = None
        self.std = None

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.mean = x.mean()
            self.std = x.std()
        else:
            # compute channel-wise mean
            y = _transform_to_ch_axis(x, self.ch_axis)
            self.mean = y.mean(1)
            self.std = y.std(1)
        self.min_val = self.mean - 3 * self.std
        self.max_val = self.mean + 3 * self.std


class AvgMinMaxObserver(ObserverBase):
    """average min/max among batches. for ptq calibration"""

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMinMaxObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        self.cnt = 0

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        assert self.ch_axis == -1
        min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class EMAMinMaxObserver(ObserverBase):
    """Moving average min/max among batches. collect statistics during training"""

    def __init__(self, bit=8, symmetric=False, ch_axis=-1, ema_ratio=0.9):
        super(EMAMinMaxObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        self.ema_ratio = ema_ratio

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        assert self.ch_axis == -1
        min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (
                1 - self.ema_ratio
            )
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (
                1 - self.ema_ratio
            )


class AvgTokenQuantileObserver(ObserverBase):
    """average min/max among batches."""

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgTokenQuantileObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        self.cnt = 0
        self.percentile = 1.0

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
            x = self.prune_token(x)
        elif seq_pos != -1:
            x = self.reshape_batch_embedding(x, seq_pos)
            x = self.prune_token(x)
        assert self.ch_axis == -1
        min_val_cur, max_val_cur = torch._aminmax(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class EMAQuantileObserver(ObserverBase):
    """Moving average quantile among batches."""

    def __init__(
        self,
        bit=8,
        symmetric=False,
        ch_axis=-1,
        ema_ratio=0.9,
        threshold=0.9999,
        bins=2048,
    ):
        super(EMAQuantileObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        assert (
            self.ch_axis == -1
        ), "Quantile observer only support in per-tensor scheme."
        self.ema_ratio = ema_ratio
        self.threshold = threshold
        self.bins = bins

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        min_val_cur, max_val_cur = torch._aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(torch.abs(x), bins=self.bins, min=0.0, max=max_hist_range)
        cur_total = 0
        clip_value = max_hist_range
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * self.ema_ratio + max(
                min_val_cur, -clip_value
            ) * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + min(
                max_val_cur, clip_value
            ) * (1.0 - self.ema_ratio)
        return x


class AvgQuantileObserver(ObserverBase):
    """Moving average quantile among batches."""

    def __init__(
        self,
        bit=8,
        symmetric=False,
        ch_axis=-1,
        ema_ratio=0.9,
        threshold=0.999,
        bins=2048,
    ):
        super(AvgQuantileObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        assert (
            self.ch_axis == -1
        ), "Quantile observer only support in per-tensor scheme."
        self.ema_ratio = ema_ratio
        self.threshold = threshold
        self.bins = bins
        self.cnt = 0

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        min_val_cur, max_val_cur = torch._aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(torch.abs(x), bins=self.bins, min=0.0, max=max_hist_range)
        cur_total = 0
        clip_value = max_hist_range
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_hist_range / self.bins)
                break
            cur_total += cnt
        min_val_cur = max(min_val_cur, -clip_value)
        max_val_cur = min(max_val_cur, clip_value)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.cnt + min_val_cur
            self.max_val = self.max_val * self.cnt + max_val_cur
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt
        return x


class MSEObserver(ObserverBase):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MSEObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.p = 2.0
        self.num = 100  # candidate num
        self.one_side_dist = None  # 'pos', 'neg', 'no'

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if self.ch_axis == -1:
            return x.mean()
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            return y.mean(1)

    def loss_fx(self, x, new_min, new_max):
        # should also consider channel here
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        if self.ch_axis != -1:
            x_q = fake_quantize_per_channel_affine(
                x,
                scale.data,
                zero_point.data.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )
        else:
            x_q = fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()), self.quant_min, self.quant_max
            )
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def perform_2D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / float(self.quant_max - self.quant_min)
            # enumerate zp
            for zp in range(self.quant_min, self.quant_max + 1):
                new_min = torch.max(tmp_min - zp * tmp_delta, x_min)
                new_max = torch.min(tmp_max - zp * tmp_delta, x_max)
                score = self.loss_fx(x, new_min, new_max)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.ch_axis != -1:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == "pos" else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == "neg" else thres
            score = self.loss_fx(x, new_min, new_max)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )

        if (
            self.one_side_dist != "no" or self.symmetric
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class AvgMSEObserver(MSEObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(AvgMSEObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )

        if (
            self.one_side_dist != "no" or self.symmetric
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.cnt + best_min
            self.max_val = self.max_val * self.cnt + best_max
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class MSEFastObserver(ObserverBase):
    # golden section search here
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MSEFastObserver, self).__init__(
            bit=bit, symmetric=symmetric, ch_axis=ch_axis
        )
        self.p = 2.0
        self.num = 100  # candidate num
        self.one_side_dist = None  # 'pos', 'neg', 'no'

    def lp_loss(self, pred, tgt, p=2.0):
        return (pred - tgt).abs().pow(p).mean()

    def loss_fx(self, x, new_min, new_max):
        # only consider tensor here
        new_min = torch.tensor(new_min).cuda()
        new_max = torch.tensor(new_max).cuda()
        scale, zero_point = self.calculate_qparams(new_min, new_max)
        x_q = fake_quantize_per_tensor_affine(
            x, scale.item(), int(zero_point.item()), self.quant_min, self.quant_max
        )
        score = self.lp_loss(x_q, x, p=self.p)
        return score

    def golden_asym_shift_loss(self, shift, xrange, x, x_min, x_max):
        tmp_min = 0.0
        tmp_max = xrange
        new_min = max(tmp_min - shift, x_min)
        new_max = min(tmp_max - shift, x_max)
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_asym_range_loss(self, xrange, x, x_min, x_max):
        tmp_delta = xrange / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        result = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(xrange, x, x_min, x_max),
            bounds=(min_shift, max_shift),
            method="Bounded",
        )
        return result.fun

    def golden_sym_range_loss(self, xrange, x):
        new_min = 0.0 if self.one_side_dist == "pos" else -xrange
        new_max = 0.0 if self.one_side_dist == "neg" else xrange
        return self.loss_fx(x, new_min, new_max).cpu().numpy()

    def golden_section_search_2D_channel(self, x, x_min, x_max):
        xrange = x_max - x_min
        result = minimize_scalar(
            self.golden_asym_range_loss,
            args=(x, x_min, x_max),
            bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()),
            method="Bounded",
        )
        final_range = result.x
        tmp_min = 0.0
        tmp_max = final_range
        tmp_delta = final_range / float(self.quant_max - self.quant_min)
        max_shift = tmp_delta * self.quant_max
        min_shift = tmp_delta * self.quant_min
        subresult = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(final_range, x, x_min, x_max),
            bounds=(min_shift, max_shift),
            method="Bounded",
        )
        final_shift = subresult.x
        best_min = max(tmp_min - final_shift, x_min)
        best_max = min(tmp_max - final_shift, x_max)
        return torch.tensor(best_min).cuda(), torch.tensor(best_max).cuda()

    def golden_section_search_1D_channel(self, x, x_min, x_max):
        xrange = torch.max(x_min.abs(), x_max)
        result = minimize_scalar(
            self.golden_sym_range_loss,
            args=(x,),
            bounds=(min(0.1, 0.01 * xrange.item()), xrange.item()),
            method="Bounded",
        )
        final_range = result.x
        best_min = (
            torch.zeros_like(x_min)
            if self.one_side_dist == "pos"
            else -torch.tensor(final_range)
        )
        best_max = (
            torch.zeros_like(x_max)
            if self.one_side_dist == "neg"
            else torch.tensor(final_range)
        )
        return torch.tensor(best_min).cuda(), torch.tensor(best_max).cuda()

    def golden_section_2D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_2D_channel(x, x_min, x_max)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(y):
                x_min[ch], x_max[ch] = self.golden_section_search_2D_channel(
                    y[ch], x_min[ch], x_max[ch]
                )
        return x_min, x_max

    def golden_section_1D_search(self, x):
        if self.ch_axis == -1:
            x_min, x_max = torch._aminmax(x)
            x_min, x_max = self.golden_section_search_1D_channel(x, x_min, x_max)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            x_min, x_max = torch._aminmax(y, 1)
            for ch, val in enumerate(y):
                x_min[ch], x_max[ch] = self.golden_section_search_1D_channel(
                    y[ch], x_min[ch], x_max[ch]
                )
        return x_min, x_max

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )

        if (
            self.one_side_dist != "no" or self.symmetric
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.golden_section_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.golden_section_2D_search(x)
        self.min_val = torch.min(self.min_val, best_min)
        self.max_val = torch.max(self.max_val, best_max)


class AvgMSEFastObserver(MSEFastObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.cnt = 0
        assert self.ch_axis == -1

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )
        if (
            self.one_side_dist != "no" or self.symmetric
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.golden_section_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.golden_section_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.cnt + best_min
            self.max_val = self.max_val * self.cnt + best_max
        self.cnt += 1
        self.min_val /= self.cnt
        self.max_val /= self.cnt


class EMAMSEFastObserver(MSEFastObserver):
    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.ema_ratio = 0.9

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )

        if (
            self.one_side_dist != "no" or self.symmetric
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.golden_section_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.golden_section_2D_search(x)
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = best_min
            self.max_val = best_max
        else:
            self.min_val = self.min_val * self.ema_ratio + best_min * (
                1 - self.ema_ratio
            )
            self.max_val = self.max_val * self.ema_ratio + best_max * (
                1 - self.ema_ratio
            )
