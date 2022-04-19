import os
import sys
sys.path.insert(0,os.getcwd())

from torchsummary import summary
import torchvision as tv
import torch

from models.cspnet.cspdarknet50 import model_cfg
from models.build import BuildNet
from thop import profile

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ = BuildNet(model_cfg.copy())
    # model = tv.models.mobilenet_v3_small()
    # input = torch.randn(1, 3, 224, 224)
    #summary(model,(3, 224, 224),1)
    summary(model_.to(device),(3, 224, 224),1)
    #input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model_, (input,))
    # print('flops: ', flops, 'params: ', params)
