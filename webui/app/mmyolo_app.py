
import os

os.system("pip install -U openmim")
os.system("mim install 'mmengine>=0.6.0'")
os.system("mim install 'mmcv>=2.0.0rc4,<2.1.0'")
os.system("mim install 'mmyolo'")


import PIL.Image
import gradio as gr
from argparse import Namespace
from pathlib import Path

import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes
from mim import download


def download_test_image():
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266264420-21575a83-4057-41cf-8a4a-b3ea6f332d79.jpg',
        'bus.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266264536-82afdf58-6b9a-4568-b9df-551ee72cb6d9.jpg',
        'dogs.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266264600-9d0c26ca-8ba6-45f2-b53b-4dc98460c43e.jpg',
        'zidane.jpg')

def download_cfg_checkpoint():
    download(package='mmyolo',
             configs=['s'],
             # configs=['yolov5_s-v61_syncbn_fast_8xb16-300e_coco'],
             dest_root='.')

def detect_objects(args):
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            show_data_classes(dataset_classes)
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        progress_bar.update()

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)

def object_detection(img_path, config, checkpoint, out_dir, device, show, score_thr, class_name):
    args = Namespace(
        img=img_path,
        config=config,
        checkpoint=checkpoint,
        out_dir=out_dir,
        device=device,
        show=show,
        score_thr=score_thr,
        class_name=class_name,
    )
    detect_objects(args)
    img_src = PIL.Image.open(img_path)
    img_out = PIL.Image.open(os.path.join(out_dir, os.path.basename(img_path)))
    return img_src, img_out, out_dir

download_test_image()
download_cfg_checkpoint()

inputs = [
    gr.inputs.Textbox(default="bus.jpg", label="img-dir"),
    gr.inputs.Textbox(
        default="yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py", label="config"),
    gr.inputs.Textbox(
        default="yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth", label="checkpoint"),
    gr.inputs.Textbox(default="./output", label="output"),
    gr.inputs.Radio(["cuda:0", "cpu"], default="cpu", label="device"),
    gr.inputs.Checkbox(default=False, label="show"),
    gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.3, label="score_thr"),
    gr.inputs.Textbox(default=None, label="class_name"),
]

examples = [
    ['bus.jpg', 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py',
     'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth', './output', "cpu", False, 0.3,None],
    ['dogs.jpg', 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py',
     'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth', './output', "cpu", False, 0.3,None],
    ['zidane.jpg','yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py',
     'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth', './output', "cpu", False, 0.3,None]
]

text_output = gr.outputs.Textbox(label="输出路径")
src_image = gr.outputs.Image(type="pil")
output_image = gr.outputs.Image(type="pil")

title = "MMYOLO detection web demo"
description = "use mmyolo detection"
article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmdetection'>MMDetection</a> 是一个开源的物体检测工具箱，提供了丰富的检测模型和数据增强方式，本项目基于 MMDetection 实现 Faster R-CNN 检测算法。</p>"

gr.Interface(fn=object_detection, inputs=inputs, outputs=[src_image, output_image, text_output],
             examples=examples, title=title,
             description=description, article=article, allow_flagging=False).launch()
