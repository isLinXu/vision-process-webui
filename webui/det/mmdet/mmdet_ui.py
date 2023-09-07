import os

import PIL
import gradio as gr
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer


# def detect_objects(call_args, img_path, model, weights, out_dir, texts, device, pred_score_thr, batch_size,
#                    show, no_save_vis, no_save_pred,
#                    print_result, palette, custom_entities):
#     call_args['inputs'] = img_path
#     call_args['model'] = model
#     call_args['weights'] = weights
#     call_args['inputs'] = img_path
#     call_args['device'] = device
#     return call_args

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
             'and alias defined in metafile. The model configuration '
             'file will try to read from .pth if the parameter is '
             'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
             'If so, the input text should be '
             '"cls_name1 . cls_name2 . cls_name3 ." format')

    call_args = vars(parser.parse_args())
    return call_args


def detect_objects(img_path, model, weights, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis,
                   no_save_pred,
                   print_result, palette, custom_entities):
    call_args = parse_args()
    call_args['model'] = model
    call_args['weights'] = weights
    call_args['inputs'] = img_path
    call_args['device'] = device
    call_args['out_dir'] = out_dir
    call_args['texts'] = texts
    call_args['pred_score_thr'] = pred_score_thr
    call_args['batch_size'] = batch_size
    call_args['show'] = show
    call_args['no_save_vis'] = no_save_vis
    call_args['no_save_pred'] = no_save_pred
    call_args['print_result'] = print_result
    call_args['palette'] = palette
    call_args['custom_entities'] = custom_entities

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main(inputs, model, weights, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis, no_save_pred,
         print_result, palette, custom_entities):
    img_path = "input_img.jpg"
    inputs.save("input_img.jpg")
    init_args, call_args = detect_objects(img_path, model, weights, out_dir, texts, device, pred_score_thr, batch_size,
                                          show, no_save_vis, no_save_pred,
                                          print_result, palette, custom_entities)

    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    inferencer = DetInferencer(**init_args)
    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')
    out_dir = './outputs/vis/'
    print("inputs:", img_path)
    img_out = PIL.Image.open(os.path.join(out_dir, img_path))
    return img_out


if __name__ == '__main__':
    iface = gr.Interface(
        fn=main,
        inputs=[
            gr.inputs.Image(type="pil", label="input"),
            gr.inputs.Textbox(label="model", default="rtmdet_tiny_8xb32-300e_coco.py"),
            gr.inputs.Textbox(label="weights", default="rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"),
            gr.inputs.Textbox(label="out_dir"),
            gr.inputs.Textbox(label="texts"),
            gr.inputs.Textbox(label="device", default="cpu"),
            gr.inputs.Slider(label="pred_score_thr", minimum=0.0, maximum=1.0, step=0.1),
            gr.inputs.Number(label="batch_size", default=1),
            gr.inputs.Checkbox(label="show"),
            gr.inputs.Checkbox(label="no_save_vis"),
            gr.inputs.Checkbox(label="no_save_pred"),
            gr.inputs.Checkbox(label="print_result"),
            gr.inputs.Radio(label="palette", choices=["coco", "voc", "citys", "random", "none"]),
            gr.inputs.Checkbox(label="custom_entities")
        ],
        outputs=gr.outputs.Image(type="pil")
    )

    iface.launch()
