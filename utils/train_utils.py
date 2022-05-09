import os
import torch
import sys
import types
import importlib
from tqdm import tqdm
from numpy import mean
from terminaltables import AsciiTable
from configs import backbones
from core.evaluations import evaluate

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
训练
'''
def train(model, runner, lr_update_func, device, epoch, epoches, save_dir, train_history):
    train_loss = 0
    
    model.train()
    with tqdm(total=len(runner.get('train_loader')),desc=f'Train: Epoch {epoch + 1}/{epoches}',postfix=dict,mininterval=0.3) as pbar:
        for iter, batch in enumerate(runner.get('train_loader')):
            runner['iter'] += 1
            images, targets = batch
            with torch.no_grad():
                images  = images.to(device)
                targets = targets.to(device)

            runner.get('optimizer').zero_grad()
            lr_update_func.before_train_iter(runner)
            losses = model(images,targets=targets,return_loss=True)
            losses.get('loss').backward()
            runner.get('optimizer').step()

            train_loss += losses.get('loss').item()
            pbar.set_postfix(**{'Loss': train_loss / (iter + 1), 
                                'Lr' : get_lr(runner.get('optimizer'))
                                })
            pbar.update(1)
    
    train_history.update(train_loss / (iter + 1),'train')
            
    if train_loss/len(runner.get('train_loader')) < runner.get('best_train_loss') :
        runner['best_train_loss'] = train_loss/len(runner.get('train_loader'))
        if epoch > 0:
            os.remove(runner['best_train_weight'])
        runner['best_train_weight'] = os.path.join(save_dir,'Train_Epoch{:03}-Loss{:.3f}.pth'.format(epoch+1,train_loss / len(runner.get('train_loader'))))
        torch.save(model.state_dict(),runner.get('best_train_weight'))
    
    if epoch > 0:
        os.remove(runner['last_weight'])
    runner['last_weight'] = os.path.join(save_dir,'Last_Epoch{:03}.pth'.format(epoch+1))
    torch.save(model.state_dict(),runner.get('last_weight'))
    

def validation(model, runner, cfg, device, epoch, epoches, save_dir, train_history):

    preds,targets = [],[]
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(runner.get('val_loader')), desc=f'Test : Epoch {epoch + 1}/{epoches}',mininterval=0.3) as pbar:
            for iter, batch in enumerate(runner.get('val_loader')):
                images, target = batch
                #runner.get('optimizer').zero_grad()
                outputs = model(images.to(device),return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))   
                pbar.update(1)
                
    eval_results = evaluate(torch.cat(preds),torch.cat(targets),cfg.get('metrics'),cfg.get('metric_options'))
    
    train_history.update(eval_results,'test')
    
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
             
    if save_dir and eval_results.get('accuracy_top-1') > runner.get('best_val_acc'):
        runner['best_val_acc'] = eval_results.get('accuracy_top-1')
        if epoch > 0:
            os.remove(runner['best_val_weight'])
        runner['best_val_weight'] = os.path.join(save_dir,'Val_Epoch{:03}-Acc{:.3f}.pth'.format(epoch+1,eval_results.get('accuracy_top-1')))
        torch.save(model.state_dict(),runner.get('best_val_weight'))
