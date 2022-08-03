from random import shuffle
from PIL import Image
import copy
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from core.datasets.compose import Compose

class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels   = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        
        return len(self.gt_labels)

    # def __getitem__(self, index):
    #     image_path = self.gt_labels[index].split(' ')[0].split()[0]
    #     image = Image.open(image_path)
    #     cfg = copy.deepcopy(self.cfg)
    #     image = self.preprocess(image,cfg)
    #     gt = int(self.gt_labels[index].split(' ')[1])
        
    #     return image, gt, image_path
    
    # def preprocess(self, image,cfg):
    #     if not (len(np.shape(image)) == 3 and np.shape(image)[2] == 3):
    #         image = image.convert('RGB')
    #     funcs = []

    #     for func in cfg:
    #         funcs.append(eval('transforms.'+func.pop('type'))(**func))
    #     image = transforms.Compose(funcs)(image)
    #     return image

    def __getitem__(self, index):
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))
        return results['img'], int(results['gt_label']), results['filename']
    
    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]
        
        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

def collate(batches):
    images, gts, image_path = tuple(zip(*batches))
    images = torch.stack(images, dim=0)
    gts = torch.as_tensor(gts)
    
    return images, gts, image_path
