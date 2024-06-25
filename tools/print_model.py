import os
import sys
# from typing import Sequence
sys.path.insert(0,os.getcwd())
import argparse

import torch
# import torch.backends.cudnn as cudnn
from torch.nn.parallel import DataParallel
from utils.train_utils import  print_info, file2dict
from core.optimizers import *
from models.build import BuildNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
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
    print_info(model_cfg)

    
    # 初始化模型,详见https://www.bilibili.com/video/BV12a411772h
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Initialize the weights.')
    model = BuildNet(model_cfg)
    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))
    
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])
    
    print(model)

if __name__ == "__main__":
    main()
