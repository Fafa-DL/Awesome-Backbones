# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import cv2

from core.visualization import imshow_infos
from core.datasets.compose import Compose
from utils.checkpoint import load_checkpoint


def init_model(model, data_cfg, device='cuda:0',mode='eval'):
    """Initialize a classifier from config file.

    Returns:
        nn.Module: The constructed classifier.
    """
    
    if mode == 'train':
        if data_cfg.get('train').get('pretrained_flag') and data_cfg.get('train').get('pretrained_weights'):
            print('Loading {}'.format(data_cfg.get('train').get('pretrained_weights').split('/')[-1]))
            load_checkpoint(model,data_cfg.get('train').get('pretrained_weights'),device,False)
            
            
    elif mode =='eval':
        print('Loading {}'.format(data_cfg.get('test').get('ckpt').split('/')[-1]))
        model.eval()
        load_checkpoint(model,data_cfg.get('test').get('ckpt'),device,False)
        
    model.to(device)
    
    return model


def inference_model(model, image, val_pipeline, classes_names):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        image (str/ndarray): The image filename or loaded image.
        val_pipeline (dict): The image preprocess pipeline.
        classes_names(list): The classes of datasets.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    if isinstance(image, str):
        if val_pipeline[0]['type'] != 'LoadImageFromFile':
            val_pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=image), img_prefix=None)
    else:
        if val_pipeline[0]['type'] == 'LoadImageFromFile':
            val_pipeline.pop(0)
        data = dict(img=image, filename=None)

    pipeline = Compose(val_pipeline)
    image = pipeline(data)['img'].unsqueeze(0)
    device = next(model.parameters()).device  # model device
    
    # forward the model
    with torch.no_grad():
        scores = model(image.to(device),return_loss=False)
        pred_score,pred_label = torch.max(scores, axis=1)
        result = {'pred_label': pred_label.item(), 'pred_score': float(pred_score)}
    result['pred_class'] = classes_names[result['pred_label']]
    return result


def show_result(img,
                result,
                text_color='white',
                font_scale=0.5,
                row_width=20,
                show=False,
                fig_size=(15, 10),
                win_name='',
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or ndarray): The image to be displayed.
        result (dict): The classification results to draw over `img`.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        font_scale (float): Font scales of texts.
        row_width (int): width between each row of results on the image.
        show (bool): Whether to show the image.
            Default: False.
        fig_size (tuple): Image show figure size. Defaults to (15, 10).
        win_name (str): The window name.
        wait_time (int): How many seconds to display the image.
            Defaults to 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (ndarray): Image with overlaid results.
    """
    img = cv2.imread(img)
    img = img.copy()

    img = imshow_infos(
        img,
        result,
        text_color=text_color,
        font_size=int(font_scale * 50),
        row_width=row_width,
        win_name=win_name,
        show=show,
        fig_size=fig_size,
        wait_time=wait_time,
        out_file=out_file)

    return img

def show_result_pyplot(model,
                       img,
                       result,
                       fig_size=(15, 10),
                       title='result',
                       wait_time=0,
                       out_file=None):
    """Visualize the classification results on the image.

    Args:
        model (nn.Module): The loaded classifier.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The classification result.
        fig_size (tuple): Figure size of the pyplot figure.
            Defaults to (15, 10).
        title (str): Title of the pyplot figure.
            Defaults to 'result'.
        wait_time (int): How many seconds to display the image.
            Defaults to 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    show_result(
        img,
        result,
        show=True,
        fig_size=fig_size,
        win_name=title,
        wait_time=wait_time,
        out_file=out_file)

