import torch.nn as nn
import torch
import torch.nn.functional as F
#   nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU, nn.Sigmoid, nn.Tanh
def ReLU(inplace=True):
    return nn.ReLU(inplace=inplace)

def ReLU6(inplace=True):
    return nn.ReLU6(inplace=inplace)

def Sigmoid():
    return nn.Sigmoid()

def LeakyReLU(inplace=True):
    return nn.LeakyReLU(inplace=inplace)

def Tanh():
    return nn.Tanh()

class HSigmoid(nn.Module):
    def __init__(self, bias=3.0, divisor=6.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)

def HSwish(inplace=True):
    return nn.Hardswish(inplace=inplace)
    
class Swish(nn.Module):
    """Swish Module.

    This module applies the swish function:

    .. math::
        Swish(x) = x * Sigmoid(x)

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return F.gelu(input)