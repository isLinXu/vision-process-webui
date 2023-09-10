

import os
os.system("pip install xtcocotools>=1.12")
os.system("pip install 'mmengine>=0.6.0'")
os.system("pip install 'mmcv>=2.0.0rc4,<2.1.0'")
os.system("pip install 'mmdet>=3.0.0,<4.0.0'")
os.system("pip install 'mmpose'")

import PIL
import cv2
import mmpose
import numpy as np

import torch
from mmpose.apis import MMPoseInferencer
import gradio as gr

import warnings

warnings.filterwarnings("ignore")

mmpose_model_list = ["human", "hand", "face", "animal", "wholebody",
                     "vitpose", "vitpose-s", "vitpose-b", "vitpose-l", "vitpose-h"]


def save_image(img, img_path):
    # Convert PIL image to OpenCV image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Save OpenCV image
    cv2.imwrite(img_path, img)


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


def predict_pose(img, model_name, out_dir):
    img_path = "input_img.jpg"
    save_image(img, img_path)
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    inferencer = MMPoseInferencer(model_name, device=device)
    result_generator = inferencer(img_path, show=False, out_dir=out_dir)
    result = next(result_generator)
    save_dir = './output/visualizations/'
    out_img_path = save_dir + img_path
    print("out_img_path: ", out_img_path)
    out_img = PIL.Image.open(out_img_path)
    return out_img

download_test_image()
input_image = gr.inputs.Image(type='pil', label="Original Image")
model_name = gr.inputs.Dropdown(choices=[m for m in mmpose_model_list], label='Model')
out_dir = gr.inputs.Textbox(label="Output Directory", default="./output")
output_image = gr.outputs.Image(type="pil", label="Output Image")

examples = [
    ['zidane.jpg', 'human'],
    ['dogs.jpg', 'animal'],
]
title = "MMPose detection web demo"
description = "<div align='center'><img src='https://raw.githubusercontent.com/open-mmlab/mmpose/main/resources/mmpose-logo.png' width='450''/><div>" \
              "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmpose'>MMPose</a> MMPose 是一款基于 PyTorch 的姿态分析的开源工具箱，是 OpenMMLab 项目的成员之一。" \
              "OpenMMLab Pose Estimation Toolbox and Benchmark..</p>"
article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmpose'>MMPose</a></p>" \
          "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

iface = gr.Interface(fn=predict_pose, inputs=[input_image, model_name, out_dir], outputs=output_image,
                     examples=examples, title=title, description=description, article=article)
iface.launch()
