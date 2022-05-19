
import argparse
import os.path as osp
import os
import sys
sys.path.insert(0,os.getcwd())
import re
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim

import matplotlib.pyplot as plt
import torch.nn as nn

from utils.train_utils import file2dict
from core.optimizers import *
from torch.utils.data import DataLoader


'''
获得学习率
'''
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class SimpleModel(nn.Module):
    """simple model that do nothing in train_step."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def train_step(self, *args, **kwargs):
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a Dataset Pipeline')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--dataset-size',
        type=int,
        help='The size of the dataset. If specify, `build_dataset` will '
        'be skipped and use this size as the dataset size.')
    parser.add_argument(
        '--ngpus',
        type=int,
        default=1,
        help='The number of GPUs used in training.')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--style', type=str, default='whitegrid', help='style of plt')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The learning rate curve plot save path')
    parser.add_argument(
        '--window-size',
        default='12*7',
        help='Size of the window to display images, in format of "$W*$H".')
    args = parser.parse_args()
    if args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."

    return args


def plot_curve(lr_list, args, iters_per_epoch, by_epoch=True):
    """Plot learning rate vs iter graph."""
    try:
        import seaborn as sns
        sns.set_style(args.style)
    except ImportError:
        print("Attention: The plot style won't be applied because 'seaborn' "
              'package is not installed, please install it if you want better '
              'show style.')
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    plt.figure(figsize=(wind_w, wind_h))
    # if legend is None, use {filename}_{key} as legend

    ax: plt.Axes = plt.subplot()

    ax.plot(lr_list, linewidth=1)
    if by_epoch:
        ax.xaxis.tick_top()
        ax.set_xlabel('Iters')
        ax.xaxis.set_label_position('top')
        sec_ax = ax.secondary_xaxis(
            'bottom',
            functions=(lambda x: x / iters_per_epoch,
                       lambda y: y * iters_per_epoch))
        sec_ax.set_xlabel('Epochs')
        #  ticks = range(0, len(lr_list), iters_per_epoch)
        #  plt.xticks(ticks=ticks, labels=range(len(ticks)))
    else:
        plt.xlabel('Iters')
    plt.ylabel('Learning Rate')

    if args.title is None:
        plt.title(f'{osp.basename(args.config)} Learning Rate curve')
    else:
        plt.title(args.title)

    if args.save_path:
        plt.savefig(args.save_path)
        print(f'The learning rate graph is saved at {args.save_path}')
    plt.show()


def simulate_train(data_loader, data_cfg, optimizer_cfg, lr_config):
    # build logger, data_loader, model and optimizer
    model = SimpleModel()
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(),**optimizer_cfg)

    # build runner
    runner = dict(
        optimizer         = optimizer,
        train_loader      = data_loader,
        iter              = 0,
        epoch             = 0,
        max_epochs       = data_cfg.get('train').get('epoches'),
        max_iters         = data_cfg.get('train').get('epoches')*len(data_loader),
    )
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)
    lr_update_func.before_run(runner)
    lr_lists = []
    with tqdm(total=runner.get('max_epochs')) as pbar:
        for epoch in range(data_cfg.get('train').get('epoches')):
            runner['epoch'] = epoch + 1
            lr_update_func.before_train_epoch(runner)
            for iter, batch in enumerate(runner.get('train_loader')):
                
                runner.get('optimizer').zero_grad()
                lr_update_func.before_train_iter(runner)
                lr_lists.append(get_lr(runner.get('optimizer')))
                runner['iter'] += 1
                runner.get('optimizer').step()
                
            pbar.update(1)
            
    return lr_lists

def main():
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    # make sure save_root exists
    if args.save_path and not args.save_path.parent.exists():
        raise Exception(f'The save path is {args.save_path}, and directory '
                        f"'{args.save_path.parent}' do not exist.")

    by_epoch = True

    # prepare data loader
    batch_size = data_cfg.get('batch_size')

    if args.dataset_size is None and by_epoch:
        train_annotations   = "datas/train.txt"
        with open(train_annotations, encoding='utf-8') as f:
            train_datas = f.readlines()
        dataset_size = len(train_datas)
    else:
        dataset_size = args.dataset_size or batch_size

    fake_dataset = list(range(dataset_size))
    data_loader = DataLoader(fake_dataset, batch_size=batch_size)
    dataset_info = (f'\nDataset infos:'
                    f'\n - Dataset size: {dataset_size}'
                    f'\n - Number of GPUs: {args.ngpus}'
                    f'\n - Total batch size: {batch_size}')
    if by_epoch:
        dataset_info += f'\n - Iterations per epoch: {len(data_loader)}'

    # simulation training process
    lr_list = simulate_train(data_loader, data_cfg, optimizer_cfg, lr_config)

    plot_curve(lr_list, args, len(data_loader), by_epoch)


if __name__ == '__main__':
    main()
