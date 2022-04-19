from random import shuffle

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

import copy

class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels   = gt_labels
        self.cfg = cfg

    def __len__(self):
        
        return len(self.gt_labels)

    def __getitem__(self, index):
        image_path = self.gt_labels[index].split(' ')[0].split()[0]
        image = Image.open(image_path)
        cfg = copy.deepcopy(self.cfg)
        image = self.preprocess(image,cfg)
        gt = int(self.gt_labels[index].split(' ')[1])
        
        return image, gt


    def preprocess(self, image,cfg):
        # 确认是RGB彩色图像
        if not (len(np.shape(image)) == 3 and np.shape(image)[2] == 3):
            image = image.convert('RGB')
        funcs = []

        for func in cfg:
            funcs.append(eval('transforms.'+func.pop('type'))(**func))
        image = transforms.Compose(funcs)(image)
        return image

def collate(batches):
    images, gts = tuple(zip(*batches))
    images = torch.stack(images, dim=0)
    gts = torch.as_tensor(gts)
    
    return images, gts
