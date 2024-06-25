import os
import sys

# from typing import Sequence
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np

import torch

from utils.train_utils import file2dict
from utils.inference import init_model, inference_backbone
from models.build import BuildNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('img', help='image path')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--feature-save-path', help='image backbone feature save path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true',
        help='whether to split validation set from training set.')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # 读取配置文件获取关键字段
    args = parse_args()
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)

    # 初始化模型,详见https://www.bilibili.com/video/BV12a411772h
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Initialize the weights.')
    backbone_model = BuildNet(model_cfg).backbone
    backbone_model = init_model(backbone_model, data_cfg, device=device, mode='eval')
    # test a single image
    feature, = inference_backbone(backbone_model, args.img, val_pipeline)

    feature_save_file_name = 'backbone_feature.npy'
    feature_save_path = args.feature_save_path
    if feature_save_path is not None:
        save_path = os.path.join(feature_save_path, feature_save_file_name)
    else:
        save_path = feature_save_file_name
    np.save(save_path, feature.cpu().detach().numpy())


if __name__ == "__main__":
    main()
