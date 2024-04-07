from torch import nn
import torch.nn.functional as F


class QuantizedLayerNorm(nn.Module):
    def __init__(self, org_module):
        super(QuantizedLayerNorm, self).__init__()
        self.normalized_shape = org_module.normalized_shape
        self.eps = org_module.eps
        self.elementwise_affine = org_module.elementwise_affine
        self.weight = org_module.weight
        self.bias = org_module.bias

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.migrate = False
        self.migrate_scale = None

    def set_migrate(self, state):
        if self.migrate_scale is None:
            self.migrate = False
        else:
            self.migrate = state

    def set_migrate_scale(self, migrate_scale):
        self.migrate_scale = migrate_scale
        self.migrate = True

    def set_migrate_bias(self, migrate_bias):
        self.migrate_bias = migrate_bias
        self.migrate = True

    def forward(self, X):
        if self.migrate:
            X = X * self.migrate_scale + self.migrate_bias
        return X
