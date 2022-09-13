import argparse
import os
import time
import cv2
import sys
sys.path.insert(0,os.getcwd())

from core.datasets.compose import Compose
from utils.train_utils import file2dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a Dataset Pipeline')
    parser.add_argument('input', help='input images path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='*',
        default=['ToTensor', 'Normalize', 'ImageToTensor', 'Collect'],
        help='the pipelines to skip when visualizing')
    parser.add_argument(
        '--output-dir',
        default='',
        type=str,
        help='folder to save output pictures, if not set, do not save.')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Default train.')
    parser.add_argument(
        '--number',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--sleep',
        type=float,
        default=1,
        help='time to sleep while display every image')
    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='whether to display images in pop-up window. Default False.')
    args = parser.parse_args()

    assert args.number > 0, "'args.number' must be larger than zero."
    if args.output_dir == '' and not args.show:
        raise ValueError("if '--output-dir' and '--show' are not set, "
                         'nothing will happen when the program running.')
    return args


def main():
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    if args.phase == 'train':
        cfg = [x for x in train_pipeline if x['type'] not in args.skip_type]
    elif args.phase in ['test', 'val']:
        cfg = [x for x in val_pipeline if x['type'] not in args.skip_type]

    pipelines = Compose(cfg)
    image_files = os.listdir(args.input)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for img in image_files[:args.number]:
        data = dict(img_info=dict(filename=os.path.join(args.input, img)), img_prefix=None)
        data = pipelines(data)['img']
        if args.show:
            cv2.imshow("0",data)
            time.sleep(args.sleep)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if args.output_dir:
            cv2.imwrite(os.path.join(args.output_dir, img), data)
    
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
