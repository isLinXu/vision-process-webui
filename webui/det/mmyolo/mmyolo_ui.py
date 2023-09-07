import fnmatch
import glob
import os
import PIL.Image
import cv2
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

model_list = ['yolov5_n-v61_syncbn_fast_8xb16-300e_coco', 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco',
              'yolov5_n-p6-v62_syncbn_fast_8xb16-300e_coco', 'yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco',
              'yolov5_n-v61_fast_1xb64-50e_voc', 'yolov5_s-v61_fast_1xb64-50e_voc',
              'yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance',
              'yolov6_s_syncbn_fast_8xb32-400e_coco', 'yolov6_n_syncbn_fast_8xb32-400e_coco',
              'yolox_tiny_fast_8xb8-300e_coco',
              'yolox_s_fast_8xb8-300e_coco', 'yolox_m_fast_8xb8-300e_coco',
              'yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco',
              'yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco', 'yolox-pose_s_8xb32-300e-rtmdet-hyp_coco',
              'rtmdet_tiny_syncbn_fast_8xb32-300e_coco',
              'kd_tiny_rtmdet_s_neck_300e_coco', 'kd_s_rtmdet_m_neck_300e_coco',
              'rtmdet_s_syncbn_fast_8xb32-300e_coco',
              'rtmdet_m_syncbn_fast_8xb32-300e_coco',
              'kd_m_rtmdet_l_neck_300e_coco',
              'yolov7_tiny_syncbn_fast_8x16b-300e_coco', 'yolov7_l_syncbn_fast_8x16b-300e_coco',
              'ppyoloe_plus_s_fast_8xb8-80e_coco',
              'ppyoloe_plus_m_fast_8xb8-80e_coco',
              'yolov8_n_syncbn_fast_8xb16-500e_coco',
              'yolov8_s_syncbn_fast_8xb16-500e_coco',
              'yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco'
              ]


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

import shutil

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print(f"Clear {folder_path} successfully.")

def download_cfg_checkpoint_model_name(model_name):
    clear_folder("./checkpoint")
    download(package='mmyolo',
             configs=[model_name],
             dest_root='./checkpoint')

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


def object_detection(img, model_name, out_dir, device, show, score_thr, class_name):
    download_cfg_checkpoint_model_name(model_name)
    # config = os.path.join('./checkpoint', config)
    # checkpoint = os.path.join('./checkpoint', checkpoint)
    path = "./checkpoint"
    config = [f for f in os.listdir(path) if fnmatch.fnmatch(f, model_name + "*.py")][0]
    config = path + "/" + config

    checkpoint = [f for f in os.listdir(path) if fnmatch.fnmatch(f, model_name + "*.pth")][0]
    checkpoint = path + "/" + checkpoint

    img_path = "input_img.jpg"
    img.save("input_img.jpg")
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
    img_out = PIL.Image.open(os.path.join(out_dir, img_path))
    return img_out

inputs = [
    gr.inputs.Image(type="pil", label="input"),
    gr.inputs.Dropdown(choices=[m for m in model_list], label='Model', default='yolov5_s-v61_syncbn_fast_8xb16-300e_coco'),
    gr.inputs.Textbox(default="./output", label="output"),
    gr.inputs.Radio(["cuda:0", "cpu"], default="cpu", label="device"),
    gr.inputs.Checkbox(default=False, label="show"),
    gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.3, label="score_thr"),
    gr.inputs.Textbox(default=None, label="class_name"),
]
download_test_image()

examples = [
    ['bus.jpg', 'yolov5_n-v61_syncbn_fast_8xb16-300e_coco', './output', "cpu", False, 0.3, None],
    ['dogs.jpg', 'yolov6_s_syncbn_fast_8xb32-400e_coco', './output', "cpu", False, 0.3, None],
    ['zidane.jpg', 'rtmdet_tiny_syncbn_fast_8xb32-300e_coco', './output', "cpu", False, 0.3, None]
]

text_output = gr.outputs.Textbox(label="输出路径")
src_image = gr.outputs.Image(type="pil")
output_image = gr.outputs.Image(type="pil")

title = "MMYOLO detection web demo"
description = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmyolo'>MMYOLO</a> 是一个开源的物体检测工具箱，提供了丰富的检测模型和数据增强方式。" \
              "OpenMMLab YOLO series toolbox and benchmark. Implemented RTMDet, RTMDet-Rotated,YOLOv5, YOLOv6, YOLOv7, YOLOv8,YOLOX, PPYOLOE, etc.</p>"
article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmyolo'>MMDetection</a></p>"

gr.Interface(fn=object_detection, inputs=inputs, outputs=output_image,
             examples=examples,
             title=title,
             description=description, article=article, allow_flagging=False).launch()
