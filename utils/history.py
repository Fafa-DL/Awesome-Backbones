from matplotlib import pyplot as plt
from numpy import mean
import os
import csv
#from torch.utils.tensorboard import SummaryWriter

class History():
    def __init__(self, dir):
        self.csv_dir = os.path.join(dir,'metrics_outputs.csv')
        self.pic_dir = os.path.join(dir,'loss-acc.png')
        self.losses_iter = []
        self.losses_epoch = []
        
        self.acc_iter = []
        self.acc_epoch = []
        
        # self.f1_epoch = []
        # self.recall_epoch = []
        # self.precision_epoch = []
        self.epoch_outputs = [['Epoch', 'Train Loss', 'Val Acc', 'Precision', 'Recall', 'F1 Score']]
        
        self.temp_data = []
        
    def update(self,data,mode):
        if mode == 'train':
            self.temp_data.append(data)
            self.losses_epoch.append(data)
        elif mode == 'test':
            self.temp_data.extend([data.get('accuracy_top-1'),mean(data.get('precision',0.0)),mean(data.get('recall',0.0)),mean(data.get('f1_score',0.0))])
            self.acc_epoch.append(data.get('accuracy_top-1'))
        
    def after_iter(self,loss,acc):
        pass
    
    def after_epoch(self,epoch):
        '''
        保存每周期的 'Train Loss', 'Val Acc', 'Precision', 'Recall', 'F1 Score'
        '''
        with open(self.csv_dir, 'w', newline='') as f:
            writer          = csv.writer(f)
            self.temp_data.insert(0,epoch)
            self.epoch_outputs.append(self.temp_data)
            self.temp_data = []
            writer.writerows(self.epoch_outputs)

        '''
        绘制每周期Train Loss以及Validation Accuracy
        '''
        total_epoch = range(len(self.losses_epoch))

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(total_epoch, self.losses_epoch, 'red', linewidth = 2, label='Train loss')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Acc')
        ax2.plot(total_epoch, self.acc_epoch, 'blue', linewidth = 2, label='Val acc')
        fig.legend()
        fig.tight_layout()
        plt.savefig(self.pic_dir)
        plt.close("all")
        
    def after_run(self):
        pass
