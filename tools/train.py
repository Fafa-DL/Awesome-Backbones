import os
import sys
# from typing import Sequence
sys.path.insert(0,os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
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
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)

    # 初始化
    meta = dict()
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs',model_cfg.get('backbone').get('type'),dirname)
    meta['save_dir'] = save_dir
    
    # 设置随机数种子
    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    meta['seed'] = seed
    
    # 读取训练&制作验证标签数据
    total_annotations   = "datas/train.txt"
    with open(total_annotations, encoding='utf-8') as f:
        total_datas = f.readlines()
    if args.split_validation:
        total_nums = len(total_datas)
        # indices = list(range(total_nums))
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            rng.shuffle(total_datas)
        val_nums = int(total_nums * args.ratio)
        folds = list(range(int(1.0 / args.ratio)))
        fold = random.choice(folds)
        val_start = val_nums * fold
        val_end = val_nums * (fold + 1)
        train_datas = total_datas[:val_start] + total_datas[val_end:]
        val_datas = total_datas[val_start:val_end]
    else:
        train_datas = total_datas.copy()
        test_annotations    = 'datas/test.txt'
        with open(test_annotations, encoding='utf-8') as f:
            val_datas   = f.readlines()
    
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
        model = DataParallel(model,device_ids=[args.gpu_id])
    
    # 初始化优化器
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(),**optimizer_cfg)
    
    # 初始化学习率更新策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)
    
    # 制作数据集->数据增强&预处理,详见https://www.bilibili.com/video/BV1zY4y167Ju
    train_dataset = Mydataset(train_datas, train_pipeline)
    val_pipeline = copy.deepcopy(train_pipeline)
    val_dataset = Mydataset(val_datas, val_pipeline)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'),pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True,
    drop_last=True, collate_fn=collate)
    
    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer         = optimizer,
        train_loader      = train_loader,
        val_loader        = val_loader,
        iter              = 0,
        epoch             = 0,
        max_epochs       = data_cfg.get('train').get('epoches'),
        max_iters         = data_cfg.get('train').get('epoches')*len(train_loader),
        best_train_loss   = float('INF'),
        best_val_acc     = float(0),
        best_train_weight = '',
        best_val_weight   = '',
        last_weight       = ''
    )
    meta['train_info'] = dict(train_loss = [],
                              val_loss = [],
                              train_acc = [],
                              val_acc = [])
    
    # 是否从中断处恢复训练
    if args.resume_from:
        model,runner,meta = resume_model(model,runner,args.resume_from,meta)
    else:
        os.makedirs(save_dir)
        shutil.copyfile(args.config,os.path.join(save_dir,os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')
        
    # 初始化保存训练信息类
    train_history =History(meta['save_dir'])
    
    # 记录初始学习率，详见https://www.bilibili.com/video/BV1WT4y1q7qN
    lr_update_func.before_run(runner)
    
    # 训练
    for epoch in range(runner.get('epoch'),runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        train(model,runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), meta)
        validation(model,runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)
        
        train_history.after_epoch(meta)

if __name__ == "__main__":
    main()
