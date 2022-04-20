# from configs.common import *
from configs.backbones import *
from configs.necks import *
from configs.heads import *
from configs.common import BaseModule,Sequential

import torch.nn as nn
import torch

import functools
from inspect import getfullargspec
from collections import abc
import numpy as np

def build_model(cfg):
    if isinstance(cfg, list):
        modules = [
            eval(cfg_.pop("type"))(**cfg_) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return eval(cfg.pop("type"))(**cfg)
    

class BuildNet(BaseModule):
    def __init__(self,cfg):
        super(BuildNet, self).__init__()
        self.neck_cfg = cfg.get("neck")
        self.head_cfg = cfg.get("head")
        self.backbone = build_model(cfg.get("backbone"))
        if self.neck_cfg is not None:
            self.neck = build_model(cfg.get("neck"))
        
        if self.head_cfg is not None:
            self.head = build_model(cfg.get("head"))

    def freeze_layers(self,names):
        assert isinstance(names,tuple)
        for name in names:
            layers = getattr(self, name)
            # layers.eval()
            for param in layers.parameters():
                param.requires_grad = False
    
    def extract_feat(self, img, stage='neck'):
        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmcv import Config
            >>> from mmcls.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(img)

        if stage == 'backbone':
            return x

        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        if stage == 'neck':
            return x
    
    def forward(self, x,return_loss=True,**kwargs):
        if return_loss:
            return self.forward_train(x,**kwargs)
        else:
            return self.forward_test(x,**kwargs)

    def forward_train(self,x,targets,**kwargs):
        x = self.extract_feat(x)
        
        losses = dict()
        loss = self.head.forward_train(x,targets,**kwargs)
        losses.update(loss)
        return losses
        
    def forward_test(self, x,**kwargs):
        x = self.extract_feat(x)
        
        out = self.head.simple_test(x,**kwargs)
        return out
    
