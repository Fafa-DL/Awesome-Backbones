import argparse
import os
import sys
sys.path.insert(0,os.getcwd())

import torch
import onnx
import onnxsim

from models.build import BuildNet
from utils.train_utils import file2dict
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Export ONNX Model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='the checkpoint file')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument('--opset', type=int, default=12, help='onnx opset version')
    args = parser.parse_args()
    return args

def main():
    # 读取配置文件获取关键字段
    args = parse_args()
    
    if len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = args.shape
    else:
        raise ValueError('invalid input shape')
    
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    
    print('Initialize the weights.')
    model = BuildNet(model_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)

    print('Exporting onnx model.')
    onnx_path = args.checkpoint.replace('.pth', '.onnx')
    torch.onnx.export(
        model, 
        (torch.randn(1, 3, *input_shape), False, False),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True
    )
    model, ok = onnxsim.simplify(onnx.load(onnx_path))
    if not ok:
        raise RuntimeError("Onnx simplifying failed.")
    onnx.save(model, onnx_path)

if __name__ == "__main__":
    main()
