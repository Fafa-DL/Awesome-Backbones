# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import sys
sys.path.insert(0,os.getcwd())

from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--save-path',
        help='The path to save prediction image, default not to save.')
    args = parser.parse_args()

    classes_map = 'datas/annotations.txt' 
    classes_names, _ = get_info(classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    model = init_model(model_cfg, data_cfg, device=args.device, mode='eval')
    # test a single image
    result = inference_model(model, args.img, val_pipeline, classes_names)
    # show the results
    show_result_pyplot(model, args.img, result, out_file=args.save_path)


if __name__ == '__main__':
    main()
