import torch
import torch.nn as nn
from .observer import MinMaxObserver
from .quant_utils import (
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
    quantize_per_channel_affine,
    dequantize_per_channel_affine,
    quantize_per_tensor_affine,
    dequantize_per_tensor_affine,
)


class QuantizeBase(nn.Module):
    def __init__(self, observer=MinMaxObserver, bit=8, symmetric=False, ch_axis=-1):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        self.name = name

    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    def disable_observer(self):
        self.observer_enabled = 0

    def enable_observer(self):
        self.observer_enabled = 1

    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    def enable_fake_quant(self):
        self.fake_quant_enabled = 1

    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, "
            "symmetric={}, bit={}, ch_axis={}, quant_min={}, quant_max={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.symmetric,
                self.bit,
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )
        )


class FixedFakeQuantize(QuantizeBase):
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if self.observer_enabled == 1:
            self.observer(
                X.detach(), observation_mask=observation_mask, seq_pos=seq_pos
            )
            _scale, _zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X,
                    self.scale.data,
                    self.zero_point.data.int(),
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                )
            else:
                X = fake_quantize_per_tensor_affine(
                    X,
                    self.scale.item(),
                    self.zero_point.item(),
                    self.quant_min,
                    self.quant_max,
                )
        return X


class GroupFixedFakeQuantize(QuantizeBase):
    # weight and activation
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, group_size=128):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.group_size = group_size
        assert type(self.observer) == MinMaxObserver
        assert ch_axis == 0

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if not self.observer_enabled and not self.fake_quant_enabled:
            return X
        org_shape = X.shape
        assert org_shape[-1] % self.group_size == 0
        X = X.reshape(-1, self.group_size)
        if self.observer_enabled == 1 or self.fake_quant_enabled == 1:
            self.observer(X.detach())
            scale, zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )

        if self.fake_quant_enabled == 1:
            X = fake_quantize_per_channel_affine(
                X,
                scale.data,
                zero_point.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )
        return X.reshape(org_shape)


class TokenGroupFixedFakeQuantize(QuantizeBase):
    # weight and activation
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, group_size=128):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.group_size = group_size
        assert type(self.observer) == MinMaxObserver
        assert ch_axis == 0

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if not self.observer_enabled and not self.fake_quant_enabled:
            return X
        org_shape = X.shape
        X = X.reshape(-1, org_shape[-1])
        assert org_shape[-1] % self.group_size == 0
        X = X.t().reshape(org_shape[-1] // self.group_size, -1)
        if self.observer_enabled == 1 or self.fake_quant_enabled == 1:
            self.observer(X.detach())
            scale, zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )

        if self.fake_quant_enabled == 1:
            X = fake_quantize_per_channel_affine(
                X,
                scale.data,
                zero_point.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )
        return X.reshape(org_shape[-1], -1).t().reshape(org_shape)


class TokenFixedFakeQuantize(QuantizeBase):
    # act: ch_axis = 0
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        # assert type(self.observer) == MinMaxObserver
        assert ch_axis == 0

    def forward(self, X, observation_mask=None, seq_pos=-1):
        if not self.observer_enabled and not self.fake_quant_enabled:
            return X
        org_shape = X.shape
        X = X.reshape(-1, org_shape[-1])
        if self.observer_enabled == 1 or self.fake_quant_enabled == 1:
            self.observer(X.detach())
            scale, zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )

        if self.fake_quant_enabled == 1:
            X = fake_quantize_per_channel_affine(
                X,
                scale.data,
                zero_point.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )
        return X.reshape(org_shape)


class FixedQuantize(QuantizeBase):
    # This is only for weight
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.is_quantized = False
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))

    def forward(self, X, observation_mask=None, seq_pos=-1):
        X_clone = X.detach().clone()
        if self.observer_enabled == 1 and not self.is_quantized:
            self.observer(X_clone)
            _scale, _zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

            if self.ch_axis != -1:
                X.data = quantize_per_channel_affine(
                    X_clone,
                    self.scale.data,
                    self.zero_point.data.int(),
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                )
            else:
                X.data = quantize_per_tensor_affine(
                    X_clone,
                    self.scale.data,
                    self.zero_point.data.int(),
                    self.quant_min,
                    self.quant_max,
                )
            X.requires_grad = False
            if self.symmetric:
                X.data = X.data.to(torch.int8)
            else:
                X.data = X.data.to(torch.uint8)
            self.is_quantized = True

        if self.fake_quant_enabled and self.is_quantized:
            X_clone = X.detach().clone()
            if self.ch_axis != -1:
                X_clone = dequantize_per_channel_affine(
                    X_clone,
                    self.scale.data,
                    self.zero_point.data.int(),
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                )
            else:
                X_clone = dequantize_per_tensor_affine(
                    X_clone,
                    self.scale.data,
                    self.zero_point.data.int(),
                    self.quant_min,
                    self.quant_max,
                )
        return X_clone


class GroupFixedQuantize(QuantizeBase):
    # This is only for weight
    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, group_size=128):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.group_size = group_size
        self.is_quantized = False
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))

    def forward(self, X, observation_mask=None, seq_pos=-1):
        X_clone = X.detach().clone()
        org_shape = X.shape
        if self.observer_enabled == 1 and not self.is_quantized:
            X_clone = X_clone.reshape(-1, self.group_size)
            self.observer(X_clone)
            _scale, _zero_point = self.observer.calculate_qparams(
                self.observer.min_val, self.observer.max_val
            )
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)
            X.data = quantize_per_channel_affine(
                X_clone,
                self.scale.data,
                self.zero_point.data.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            ).reshape(org_shape)
            X.requires_grad = False
            if self.symmetric:
                X.data = X.data.to(torch.int8)
            else:
                X.data = X.data.to(torch.uint8)
            self.is_quantized = True

        if self.fake_quant_enabled == 1 and self.is_quantized:
            X_clone = X.detach().clone()
            X_clone = X_clone.reshape(-1, self.group_size)
            X_clone = dequantize_per_channel_affine(
                X_clone,
                self.scale.data,
                self.zero_point.data.int(),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            ).reshape(org_shape)

        return X_clone
