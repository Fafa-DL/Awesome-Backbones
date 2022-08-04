from argparse import ArgumentParser
import os
import sys
sys.path.insert(0,os.getcwd())
import torch
import cv2

from utils.inference import inference_model, init_model
from core.visualization.image import imshow_infos
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='classes map of datasets')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--save-path',
        help='The path to save prediction image, default not to save.')
    parser.add_argument('--show', action='store_true', help='Show video')
    args = parser.parse_args()

    classes_names, _ = get_info(args.classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')
    
    cap = cv2.VideoCapture(args.video)
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(
            args.save_path, fourcc, fps,
            size)

    while True:
        flag, frame = cap.read()
        if not flag:
            break
        # get single test results
        result = inference_model(model, frame, val_pipeline, classes_names)
        # put the results to img
        img = imshow_infos(frame, result,show = False)

        if args.show:
            cv2.namedWindow('video', 0)
            cv2.imshow('video',img)

        if args.save_path:
            video_writer.write(img)
        
        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
