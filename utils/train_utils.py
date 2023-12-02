import os
import torch
import torch.distributed as dist
import sys
import types
import importlib
import random
from tqdm import tqdm
import numpy as np
from numpy import mean
from terminaltables import AsciiTable
from torch.optim import Optimizer
from core.evaluations import evaluate
from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.common import get_dist_info


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

'''
读取配置文件
'''
def file2dict(filename):
    (path,file) = os.path.split(filename)

    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0,abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)
                    }
    return cfg_dict.get('model_cfg'),cfg_dict.get('train_pipeline'),cfg_dict.get('val_pipeline'),cfg_dict.get('data_cfg'),cfg_dict.get('lr_config'),cfg_dict.get('optimizer_cfg')

'''
输出信息
'''
def print_info(cfg):
    backbone = cfg.get('backbone').get('type') if cfg.get('backbone') is not None else 'None'
    
    if isinstance(cfg.get('neck'),list):
        temp = []
        lists = cfg.get('neck')
        for i in lists:
            temp.append(i.get('type'))
        neck = ' '.join(temp)
    else:
        neck = cfg.get('neck').get('type') if cfg.get('neck') is not None else 'None'
        
    head = cfg.get('head').get('type') if cfg.get('head') is not None else 'None'
    loss = cfg.get('head').get('loss').get('type') if cfg.get('head').get('loss') is not None else 'None'
    
    # pretrained = os.path.basename(cfg.get('train').get('pretrained_weights')) if cfg.get('train').get('pretrained_flag') else 'None'
    # freeze = ' '.join(list(cfg.get('train').get('freeze_layers')))

    TITLE = 'Model info'
    TABLE_DATA = (
    ('Backbone', 'Neck', 'Head', 'Loss'),
    (backbone,neck,head,loss))
    
    table_instance = AsciiTable(TABLE_DATA,TITLE)
    print()
    print(table_instance.table)
    print()

'''
获得类名、索引
'''
def get_info(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name,index = data.split(' ')
        names.append(name)
        indexs.append(int(index))
        
    return names,indexs

'''
获得学习率
'''
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

'''
从中断训练处恢复训练
'''
def resume_model(model, runner, checkpoint, meta, resume_optimizer=True, map_location='default'):
    if map_location == 'default':
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(
                model,
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(model, checkpoint)
    else:
        checkpoint = load_checkpoint(
            model, checkpoint, map_location=map_location)

    runner['epoch'] = checkpoint['meta']['epoch']
    runner['iter'] = checkpoint['meta']['iter']
    runner['best_train_weight'] = checkpoint['meta']['best_train_weight']
    runner['last_weight'] = checkpoint['meta']['last_weight']
    runner['best_val_weight'] = checkpoint['meta']['best_val_weight']
    runner['best_train_loss'] = checkpoint['meta']['best_train_loss']
    runner['best_val_acc'] = checkpoint['meta']['best_val_acc']
    if meta is None:
        meta = {}

    # # Re-calculate the number of iterations when resuming
    # # models with different number of GPUs
    # if 'config' in checkpoint['meta']:
    #     config = mmcv.Config.fromstring(
    #         checkpoint['meta']['config'], file_format='.py')
    #     previous_gpu_ids = config.get('gpu_ids', None)
    #     if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
    #             previous_gpu_ids) != self.world_size:
    #         self._iter = int(self._iter * len(previous_gpu_ids) /
    #                             self.world_size)

    # resume meta information meta
    meta = checkpoint['meta']

    if 'optimizer' in checkpoint and resume_optimizer:
        if isinstance(runner['optimizer'], Optimizer):
            runner['optimizer'].load_state_dict(checkpoint['optimizer'])
        elif isinstance(runner['optimizer'], dict):
            for k in runner['optimizer'].keys():
                runner.optimizer[k].load_state_dict(
                    checkpoint['optimizer'][k])
        else:
            raise TypeError(
                'Optimizer should be dict or torch.optim.Optimizer '
                f'but got {type(runner.optimizer)}')

    print('resumed epoch %d, iter %d'% (runner['epoch'], runner['iter']))
    return model, runner, meta

'''
训练
'''
def train(model, runner, lr_update_func, device, epoch, epoches, test_cfg, meta):
    train_loss = 0
    pred_list, target_list = [], []
    runner['epoch'] = epoch + 1
    meta['epoch'] = runner['epoch']
    
    model.train()
    with tqdm(total=len(runner.get('train_loader')),desc=f'Train: Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
        for iter, batch in enumerate(runner.get('train_loader')):            
            images, targets, _ = batch
            with torch.no_grad():
                images  = images.to(device)
                targets = targets.to(device)
                target_list.append(targets)
            
            runner.get('optimizer').zero_grad()
            lr_update_func.before_train_iter(runner)
            preds, losses = model(images, targets=targets, return_loss=True, train_statu=True)
            losses.get('loss').backward()
            runner.get('optimizer').step()

            pred_list.append(preds)
            train_loss += losses.get('loss').item()
            pbar.set_postfix(**{'Loss': train_loss / (iter + 1), 
                                'Lr' : get_lr(runner.get('optimizer'))
                                })
            runner['iter'] += 1
            meta['iter'] = runner['iter']
            pbar.update(1)
    
    eval_results = evaluate(torch.cat(pred_list), torch.cat(target_list), test_cfg.get('metrics'), test_cfg.get('metric_options'))
    
    meta['train_info']['train_loss'].append(train_loss / (iter + 1))
    meta['train_info']['train_acc'].append(eval_results)
            
    if train_loss/len(runner.get('train_loader')) < runner.get('best_train_loss') :
        runner['best_train_loss'] = train_loss/len(runner.get('train_loader'))
        meta['best_train_loss'] = runner['best_train_loss']
        if epoch > 0 and os.path.isfile(runner['best_train_weight']):
            os.remove(runner['best_train_weight'])
        runner['best_train_weight'] = os.path.join(meta['save_dir'],'Train_Epoch{:03}-Loss{:.3f}.pth'.format(epoch+1,train_loss / len(runner.get('train_loader'))))
        meta['best_train_weight'] = runner['best_train_weight']
        save_checkpoint(model,runner.get('best_train_weight'),runner.get('optimizer'), meta)
    
    TITLE = 'Train Results'
    TABLE_DATA = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',100.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    print()
    

def validation(model, runner, cfg, device, epoch, epoches, meta):
    pred_list, target_list = [], []
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(runner.get('val_loader')), desc=f'Test : Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            for iter, batch in enumerate(runner.get('val_loader')):
                images, targets, _ = batch
                preds, losses = model(images.to(device), targets = targets.to(device), return_loss=True, train_statu=True)
                pred_list.append(preds)
                target_list.append(targets.to(device))  
                val_loss += losses.get('loss').item()
                pbar.set_postfix(**{'Loss': val_loss / (iter + 1)})
                pbar.update(1)
                
    eval_results = evaluate(torch.cat(pred_list),torch.cat(target_list),cfg.get('metrics'),cfg.get('metric_options'))
    
    meta['train_info']['val_acc'].append(eval_results)
    meta['train_info']['val_loss'].append(val_loss / (iter + 1))
    
    TITLE = 'Validation Results'
    TABLE_DATA = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',100.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    print()
             
    if eval_results.get('accuracy_top-1') > runner.get('best_val_acc'):
        runner['best_val_acc'] = eval_results.get('accuracy_top-1')
        meta['best_val_acc'] = runner['best_val_acc']
        if epoch > 0 and os.path.isfile(runner['best_val_weight']):
            os.remove(runner['best_val_weight'])
        runner['best_val_weight'] = os.path.join(meta['save_dir'],'Val_Epoch{:03}-Acc{:.3f}.pth'.format(epoch+1,eval_results.get('accuracy_top-1')))
        meta['best_val_weight'] = runner['best_val_weight']
        save_checkpoint(model,runner.get('best_val_weight'),runner.get('optimizer'),meta)

    if epoch > 0 and os.path.isfile(runner['last_weight']):
        os.remove(runner['last_weight'])
    runner['last_weight'] = os.path.join(meta['save_dir'],'Last_Epoch{:03}.pth'.format(epoch+1))
    meta['last_weight'] = runner['last_weight']
    save_checkpoint(model,runner.get('last_weight'),runner.get('optimizer'),meta)
