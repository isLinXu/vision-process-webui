# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from argparse import ArgumentParser

import PIL
import cv2
import gradio as gr
import numpy as np
import torch
from PIL.Image import Image
from mmocr.apis.inferencers import MMOCRInferencer

import warnings

warnings.filterwarnings("ignore")

def save_image(img, img_path):
    # Convert PIL image to OpenCV image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Save OpenCV image
    cv2.imwrite(img_path, img)


textdet_model_list = ['DBNet', 'DRRG', 'FCENet', 'PANet', 'PSENet', 'TextSnake', 'MaskRCNN']
textrec_model_list = ['ABINet', 'ASTER', 'CRNN', 'MASTER', 'NRTR', 'RobustScanner', 'SARNet', 'SATRN', 'SVTR']
textkie_model_list = ['SDMGR']


def ocr_inference(inputs, out_dir, det, det_weights, rec, rec_weights, kie, kie_weights, device, batch_size):
    init_args, call_args = parse_args()
    inputs = np.array(inputs)
    img_path = "demo_text_ocr.jpg"
    save_image(inputs, img_path)
    if det is not None and rec is not None:
        init_args['det'] = det
        init_args['det_weights'] = None
        init_args['rec'] = rec
        init_args['rec_weights'] = None
    elif det_weights is not None and rec_weights is not None:
        init_args['det'] = None
        init_args['det_weights'] = det_weights
        init_args['rec'] = None
        init_args['rec_weights'] = rec_weights
    if kie is not None:
        init_args['kie'] = kie
        init_args['kie_weights'] = None

    call_args['inputs'] = img_path
    call_args['out_dir'] = out_dir
    call_args['batch_size'] = int(batch_size)
    call_args['show'] = False
    call_args['save_pred'] = True
    call_args['save_vis'] = True
    init_args['device'] = device

    ocr = MMOCRInferencer(**init_args)
    ocr(**call_args)
    save_vis_dir = './results/vis/'
    save_pred_dir = './results/preds/'
    img_out = PIL.Image.open(os.path.join(save_vis_dir, img_path))
    json_out = json.load(open(os.path.join(save_pred_dir, img_path.replace('.jpg', '.json'))))
    return img_out, json_out


def download_test_image():
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266821429-9a897c0a-5b02-4260-a65b-3514b758f6b6.jpg',
        'demo_densetext_det.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266821432-17bb0646-a3e9-451e-9b4d-6e41ce4c3f0c.jpg',
        'demo_text_recog.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266821434-fe0d4d18-f3e2-4acf-baf5-0d2e318f0b09.jpg',
        'demo_text_ocr.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266821435-5d7af2b4-cb84-4355-91cb-37d90e91aa30.jpg',
        'demo_text_det.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266821436-4790c6c1-2da5-45c7-b837-04eeea0d7264.jpeg',
        'demo_kie.jpg')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./results/',
        help='Output directory of results.')
    parser.add_argument(
        '--det',
        type=str,
        default=None,
        help='Pretrained text detection algorithm. It\'s the path to the '
             'config file or the model name defined in metafile.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model. '
             'If it is not specified and "det" is a model name of metafile, the '
             'weights will be loaded from metafile.')
    parser.add_argument(
        '--rec',
        type=str,
        default=None,
        help='Pretrained text recognition algorithm. It\'s the path to the '
             'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected recog model. '
             'If it is not specified and "rec" is a model name of metafile, the '
             'weights will be loaded from metafile.')
    parser.add_argument(
        '--kie',
        type=str,
        default=None,
        help='Pretrained key information extraction algorithm. It\'s the path'
             'to the config file or the model name defined in metafile.')
    parser.add_argument(
        '--kie-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected kie model. '
             'If it is not specified and "kie" is a model name of metafile, the '
             'weights will be loaded from metafile.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
             'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--save_pred',
        action='store_true',
        help='Save the inference results to out_dir.')
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='Save the visualization results to out_dir.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'det', 'det_weights', 'rec', 'rec_weights', 'kie', 'kie_weights', 'device'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


if __name__ == '__main__':
    # Define Gradio input and output types
    input_image = gr.inputs.Image(type="pil", label="Input Image")
    out_dir = gr.inputs.Textbox(default="results")
    det = gr.inputs.Dropdown(label="Text Detection Model", choices=[m for m in textdet_model_list], default='DBNet')
    det_weights = gr.inputs.Textbox(default=None)
    rec = gr.inputs.Dropdown(label="Text Recognition Model", choices=[m for m in textrec_model_list], default='CRNN')
    rec_weights = gr.inputs.Textbox(default=None)
    kie = gr.inputs.Dropdown(label="Key Information Extraction Model", choices=[m for m in textkie_model_list],
                             default='SDMGR')
    kie_weights = gr.inputs.Textbox(default=None)
    device = gr.inputs.Radio(choices=["cpu", "cuda"], label="Device used for inference", default="cpu")
    batch_size = gr.inputs.Number(default=1, label="Inference batch size")
    output_image = gr.outputs.Image(type="pil", label="Output Image")
    output_json = gr.outputs.Textbox()
    download_test_image()
    examples = [["demo_text_ocr.jpg", "results", "DBNet", None, "CRNN", None, "SDMGR", None, "cpu", 1],
                ["demo_text_det.jpg", "results", "FCENet", None, "ASTER", None, "SDMGR", None, "cpu", 1],
                ["demo_text_recog.jpg", "results", "PANet", None, "MASTER", None, "SDMGR", None, "cpu", 1],
                ["demo_densetext_det.jpg", "results", "PSENet", None, "CRNN", None, "NRTR", None, "cpu", 1],
                ["demo_kie.jpg", "results", "TextSnake", None, "RobustScanner", None, "SDMGR", None, "cpu", 1]
                ]

    title = "MMOCR web demo"
    description = "<div align='center'><img src='https://raw.githubusercontent.com/open-mmlab/mmocr/main/resources/mmocr-logo.png' width='450''/><div>" \
                  "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmocr'>MMOCR</a> MMOCR 是基于 PyTorch 和 mmdetection 的开源工具箱，专注于文本检测，文本识别以及相应的下游任务，如关键信息提取。 它是 OpenMMLab 项目的一部分。" \
                  "OpenMMLab Text Detection, Recognition and Understanding Toolbox.</p>"
    article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmocr'>MMOCR</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

    # Create Gradio interface
    iface = gr.Interface(
        fn=ocr_inference,
        inputs=[
            input_image, out_dir, det, det_weights, rec, rec_weights,
            kie, kie_weights, device, batch_size
        ],
        outputs=[output_image, output_json], examples=examples,
        title=title, description=description, article=article,
    )

    # Launch Gradio interface
    iface.launch()
