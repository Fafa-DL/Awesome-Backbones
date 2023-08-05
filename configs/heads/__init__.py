from .linear_head import LinearClsHead
from .stacked_head import StackedLinearClsHead
from .cls_head import ClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .deit_head import DeiTClsHead
from .conformer_head import ConformerHead
from .efficientformer_head import EfficientFormerClsHead
from .levit_head import LeViTClsHead

__all__ = ['LinearClsHead', 'StackedLinearClsHead','ClsHead', 'VisionTransformerClsHead', 'DeiTClsHead', 'ConformerHead', 'EfficientFormerClsHead', 'LeViTClsHead']
