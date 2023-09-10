# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from argparse import ArgumentParser

import PIL
import cv2
import gradio as gr
import numpy as np
from PIL.Image import Image
from mmocr.apis.inferencers import MMOCRInferencer


def save_image(img, img_path):
    # Convert PIL image to OpenCV image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Save OpenCV image
    cv2.imwrite(img_path, img)

textdet_model_list = ['DBNet', 'DRRG', 'FCENet', 'PANet', 'PSENet', 'TextSnake', 'MaskRCNN']
textrec_model_list = []

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
    elif kie is not None:
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


def main():
    init_args, call_args = parse_args()
    ocr = MMOCRInferencer(**init_args)
    ocr(**call_args)


if __name__ == '__main__':
    # main()
    '''
    inputs, out_dir, det, det_weights, rec, rec_weights, kie, kie_weights, device, batch_size, show
    '''
    # Define Gradio input and output types
    input_image = gr.inputs.Image(type="pil", label="Input Image")
    out_dir = gr.inputs.Textbox(default="results")
    det = gr.inputs.Textbox(default="DBNet")
    det_weights = gr.inputs.Textbox(default=None)
    rec = gr.inputs.Textbox(default="CRNN")
    rec_weights = gr.inputs.Textbox(default=None)
    kie = gr.inputs.Textbox(default=None)
    kie_weights = gr.inputs.Textbox(default=None)
    device = gr.inputs.Radio(choices=["cpu", "cuda"], label="Device used for inference", default="cpu")
    batch_size = gr.inputs.Number(default=1, label="Inference batch size")
    output_image = gr.outputs.Image(type="pil", label="Output Image")
    output_json = gr.outputs.Textbox()
    # Create Gradio interface
    iface = gr.Interface(
        fn=ocr_inference,
        inputs=[
            input_image, out_dir, det, det_weights, rec, rec_weights,
            kie, kie_weights, device, batch_size
        ],
        outputs=[output_image, output_json],
        title="OCR Inference",
        description="An OCR inference app using MMOCR.",
    )

    # Launch Gradio interface
    iface.launch()
