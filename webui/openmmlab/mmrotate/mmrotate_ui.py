import fnmatch
import os

import PIL
import cv2
import gradio as gr
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mim import download
import warnings

warnings.filterwarnings("ignore")

import mmrotate

mmrorate_model_list = ['cfa_r50_fpn_1x_dota_le135', 'cfa_r50_fpn_40e_dota_oc',
                       'rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90',
                       'g_reppoints_r50_fpn_1x_dota_le135', 'gliding_vertex_r50_fpn_1x_dota_le90',
                       'rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90',
                       'rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135', 'r3det_kfiou_ln_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90', 'r3det_kld_r50_fpn_1x_dota_oc',
                       'r3det_kld_stable_r50_fpn_1x_dota_oc', 'r3det_tiny_kld_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_hbb_kld_stable_r50_fpn_6x_hrsc_rr_oc',
                       'rotated_retinanet_obb_kld_stable_r50_fpn_6x_hrsc_rr_le90',
                       'rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90',
                       'oriented_rcnn_r50_fpn_fp16_1x_dota_le90', 'oriented_rcnn_r50_fpn_1x_dota_le90',
                       'r3det_r50_fpn_1x_dota_oc', 'r3det_tiny_r50_fpn_1x_dota_oc',
                       'redet_re50_refpn_fp16_1x_dota_le90', 'redet_re50_refpn_1x_dota_le90',
                       'redet_re50_refpn_1x_dota_ms_rr_le90', 'redet_re50_refpn_3x_hrsc_le90',
                       'roi_trans_r50_fpn_fp16_1x_dota_le90', 'roi_trans_r50_fpn_1x_dota_le90',
                       'roi_trans_swin_tiny_fpn_1x_dota_le90', 'roi_trans_r50_fpn_1x_dota_ms_le90',
                       'rotated_atss_hbb_r50_fpn_1x_dota_oc', 'rotated_atss_obb_r50_fpn_1x_dota_le90',
                       'rotated_atss_obb_r50_fpn_1x_dota_le135', 'rotated_faster_rcnn_r50_fpn_1x_dota_le90',
                       'rotated_fcos_sep_angle_r50_fpn_1x_dota_le90', 'rotated_fcos_r50_fpn_1x_dota_le90',
                       'rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90', 'rotated_fcos_kld_r50_fpn_1x_dota_le90',
                       'rotated_reppoints_r50_fpn_1x_dota_oc', 'rotated_retinanet_hbb_r50_fpn_1x_dota_oc',
                       'rotated_retinanet_obb_r50_fpn_1x_dota_le90', 'rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90',
                       'rotated_retinanet_obb_r50_fpn_1x_dota_le135',
                       'rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90',
                       'rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc', 'rotated_retinanet_obb_r50_fpn_6x_hrsc_rr_le90',
                       's2anet_r50_fpn_1x_dota_le135', 's2anet_r50_fpn_fp16_1x_dota_le135',
                       'sasm_reppoints_r50_fpn_1x_dota_oc']

path = "./checkpoint"
if not os.path.exists(path):
    os.makedirs(path)

def clear_folder(folder_path):
    import shutil
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
    download(package='mmrotate',
             configs=[model_name],
             dest_root='./checkpoint')
def download_test_image():
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266800230-e8396b83-92a7-4367-bc4b-a36348e63dbe.jpg',
        'demo.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266800231-d544d5ea-fc91-45d5-b79e-97bb9c717259.jpg',
        'dota_demo.jpg')

def save_image(img, img_path):
    # Convert PIL image to OpenCV image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Save OpenCV image
    cv2.imwrite(img_path, img)


def predict_image(image, model_name, palette, score_thr, device):
    image = np.array(image)
    save_dir = './output_img.jpg'
    download_cfg_checkpoint_model_name(model_name)
    config = [f for f in os.listdir(path) if fnmatch.fnmatch(f, "*.py")][0]
    config = path + "/" + config

    checkpoint = [f for f in os.listdir(path) if fnmatch.fnmatch(f, "*.pth")][0]
    checkpoint = path + "/" + checkpoint

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    result = inference_detector(model, image)
    # show the results
    show_result_pyplot(
        model,
        image,
        result,
        palette=palette,
        score_thr=score_thr,
        out_file=save_dir)
    img_out = PIL.Image.open(save_dir)
    return img_out

download_test_image()
inputs = [
    gr.inputs.Image(type='pil', label="Input Image"),
    gr.inputs.Dropdown(label="Model Name", choices=[m for m in mmrorate_model_list], default='oriented_rcnn_r50_fpn_1x_dota_le90'),
    gr.inputs.Dropdown(label="Color palette used for visualization",
                       choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'], default='dota'),
    gr.inputs.Slider(label="bbox score threshold", minimum=0.0, maximum=1.0, step=0.01, default=0.3),
    gr.inputs.Dropdown(label="Device used for inference", choices=['cuda:0', 'cpu'], default='cpu'),
]

output = gr.outputs.Image(type='pil', label="Output Image")

title = "MMRotate detection web demo"
description = "<div align='center'><img src='https://raw.githubusercontent.com/open-mmlab/mmrotate/main/resources/mmrotate-logo.png' width='450''/><div>" \
              "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmrotate'>MMSegmentation</a> MMRotate 是一款基于 PyTorch 的旋转框检测的开源工具箱，是 OpenMMLab 项目的成员之一。" \
              "OpenMMLab Rotated Object Detection Toolbox and Benchmark.</p>"
article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmrotate'>MMRotate</a></p>" \
          "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"
examples = [["demo.jpg", "oriented_rcnn_r50_fpn_1x_dota_le90"],
            ["dota_demo.jpg", "r3det_r50_fpn_1x_dota_oc"],
            ]
gr.Interface(
    fn=predict_image,
    inputs=inputs,
    outputs=output,
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging=False,
    theme="default"
).launch()
