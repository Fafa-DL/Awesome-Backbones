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
try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.autocast would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    # Note that when PyTorch >= 1.6.0, we still cast tensor types to fp16
    # manually, so the behavior may not be consistent with real amp.
    from torch.cuda.amp import autocast
except ImportError:
    pass

def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Note:
        In v1.4.4 and later, ``cast_tersor_type`` will only convert the
        torch.Tensor which is consistent with ``src_type`` to the ``dst_type``.
        Before v1.4.4, it ignores the ``src_type`` argument, leading to some
        potential problems. For example,
        ``cast_tensor_type(inputs, torch.float, torch.half)`` will convert all
        tensors in inputs to ``torch.half`` including those originally in
        ``torch.Int`` or other types, which is not expected.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        # we need to ensure that the type of inputs to be casted are the same
        # as the argument `src_type`.
        return inputs.to(dst_type) if inputs.dtype == src_type else inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs

def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            # if (TORCH_VERSION != 'parrots' and
            #         digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
            with autocast(enabled=True):
                output = old_func(*new_args, **new_kwargs)
            # else:
            #     output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper

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
            # if name == 'backbone':
            #     for param in self.backbone.parameters():
            #         param.requires_grad = False
            # elif name == 'neck':
            #     for param in self.neck.parameters():
            #         param.requires_grad = False
            # elif name == 'head':
            #     for param in self.backbone.parameters():
            #         param.requires_grad = False
    
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
    
    #@auto_fp16()
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
    
