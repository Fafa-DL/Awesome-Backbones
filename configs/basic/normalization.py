import torch.nn as nn
import torch.nn.functional as F

def BN(*args, **kwargs):
    return nn.BatchNorm2d

def BN1d(*args, **kwargs):
    return nn.BatchNorm1d

def BN2d(*args, **kwargs):
    return nn.BatchNorm2d

def BN3d(*args, **kwargs):
    return nn.BatchNorm3d

def SyncBN(*args, **kwargs):
    return nn.SyncBatchNorm

def GN(*args, **kwargs):
    return nn.GroupNorm

def LN(*args, **kwargs):
    return nn.LayerNorm

def IN(*args, **kwargs):
    return nn.InstanceNorm2d

def IN1d(*args, **kwargs):
    return nn.InstanceNorm1d

def IN2d(*args, **kwargs):
    return nn.InstanceNorm2d

def IN3d(*args, **kwargs):
    return nn.InstanceNorm3d

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

def LN2d(*args, **kwargs):
    return LayerNorm2d