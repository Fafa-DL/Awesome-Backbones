import os
from shutil import copy, rmtree
import random
from tqdm import tqdm

def main():
    '''
    split_rate  : 测试集划分比例
    init_dataset: 未划分前的数据集路径
    new_dataset : 划分后的数据集路径
    
    '''
    def makedir(path):
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)
    
    split_rate = 0.2
    init_dataset = ''
    new_dataset = 'datasets'
    random.seed(0)

    classes_name = [name for name in os.listdir(init_dataset)]

    makedir(new_dataset)
    training_set = os.path.join(new_dataset, "train")
    test_set = os.path.join(new_dataset, "test")
    makedir(training_set)
    makedir(test_set)
    
    for cla in classes_name:
        makedir(os.path.join(training_set, cla))
        makedir(os.path.join(test_set, cla))

    
    for cla in classes_name:
        class_path = os.path.join(init_dataset, cla)
        img_set = os.listdir(class_path)
        num = len(img_set)
        test_set_index = random.sample(img_set, k=int(num*split_rate))
        with tqdm(total=num,desc=f'Class : ' + cla, mininterval=0.3) as pbar:
            for _, img in enumerate(img_set):
                if img in test_set_index:
                    init_img = os.path.join(class_path, img)
                    new_img = os.path.join(test_set, cla)
                    copy(init_img, new_img)
                else:
                    init_img = os.path.join(class_path, img)
                    new_img = os.path.join(training_set, cla)
                    copy(init_img, new_img)
                pbar.update(1)
        print()

if __name__ == '__main__':
    main()
