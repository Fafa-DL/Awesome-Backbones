import torch.nn as nn

def zero(*args, **kwargs):
    return nn.ZeroPad2d(*args, **kwargs)

def reflect(*args, **kwargs):
    return nn.ReflectionPad2d(*args, **kwargs)

def replicate(*args, **kwargs):
    return nn.ReplicationPad2d(*args, **kwargs)