
from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
import copy
from .cls_head import ClsHead
from configs.basic.activations import *
from configs.common .base_module import BaseModule, ModuleList


class LinearBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fc = nn.Linear(in_channels, out_channels)
        
        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = eval(norm_cfg.pop('type'))(**norm_cfg)
        if act_cfg is not None:
            self.act = eval(act_cfg.pop('type'))(**act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class StackedLinearClsHead(ClsHead):
    """Classifier head with several hidden fc layer and a output fc layer.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence): Number of channels in the hidden fc layers.
        dropout_rate (float): Dropout rate after each hidden fc layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each hidden fc layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 mid_channels,
                 dropout_rate: float = 0.,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(StackedLinearClsHead, self).__init__(**kwargs)
        assert num_classes > 0, \
            f'`num_classes` of StackedLinearClsHead must be a positive ' \
            f'integer, got {num_classes} instead.'
        self.num_classes = num_classes
        self.in_channels = in_channels

        assert isinstance(mid_channels, Sequence), \
            f'`mid_channels` of StackedLinearClsHead should be a sequence, ' \
            f'instead of {type(mid_channels)}'
        self.mid_channels = mid_channels
        
        self.act_cfg = act_cfg
        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        
        self._init_layers()

    def _init_layers(self):
        self.layers = ModuleList()
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            # print(self.act_cfg)
            self.layers.append(
                LinearBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=copy.deepcopy(self.act_cfg)))
            in_channels = hidden_channels

        self.layers.append(
            LinearBlock(
                self.mid_channels[-1],
                self.num_classes,
                dropout_rate=0.,
                norm_cfg=None,
                act_cfg=None))
        
    def init_weights(self):
        return self.layers.init_weights()
    
    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        for layer in self.layers[:-1]:
            x = layer(x)
        return x
    
    def fc(self,x):
        return self.layers[-1](x)
    
    def simple_test(self, x, softmax=True, post_process=False):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
    
    def forward_train(self,x,gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score,gt_label, **kwargs)

        return losses
