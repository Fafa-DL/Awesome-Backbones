import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable

import torch
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import csv

from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info,validation, file2dict
from models.build import BuildNet
from core.evaluations import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main(): 
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('eval_results',dirname)
    os.makedirs(save_dir)
    metrics_output = os.path.join(save_dir,'results.txt')
    confusion_matrix_output = os.path.join(save_dir,'confusion_matrix.csv')
    
    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_map = 'datas/cls_classes.txt' 
    test_annotations    = 'datas/test.txt'
    classes_names, indexs = get_info(classes_map)
    with open(test_annotations, encoding='utf-8') as f:
        test_datas   = f.readlines()
        
    """
    生成模型、加载权重
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BuildNet(model_cfg).to(device)
    print('Loading {}'.format(data_cfg.get('test').get('ckpt').split('/')[-1]))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(data_cfg.get('test').get('ckpt'), map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    """
    制作测试集并喂入Dataloader
    """
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True,drop_last=True, collate_fn=collate)
    
    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    model.eval()
    with torch.no_grad():
        preds,targets = [],[]
        with tqdm(total=len(test_datas)//data_cfg.get('batch_size')) as pbar:
            for _, batch in enumerate(test_loader):
                images, target = batch
                outputs = model(images.to(device),return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                pbar.update(1)
    eval_results = evaluate(torch.cat(preds),torch.cat(targets),data_cfg.get('test').get('metrics'),data_cfg.get('test').get('metric_options'))
    
    f = open(metrics_output,'w')
    
    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Clasee','Precision','Recall','F1 Score']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    f.write(table_instance.table + '\n')
    print()

    TITLE = 'Total Results'    
    TABLE_DATA_2 = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',0.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    f.write(table_instance.table + '\n')
    print()


    writer_list     = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    for i in range(len(eval_results.get('confusion'))):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3,TITLE)
    print(table_instance.table)
    f.write(table_instance.table)
    print()
    f.close()
    
    """
    以csv形式保存混淆矩阵
    """
    with open(confusion_matrix_output, 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in classes_names])
        for i in range(len(eval_results.get('confusion'))):
            writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
        writer.writerows(writer_list)
                    

if __name__ == "__main__":
    main()
