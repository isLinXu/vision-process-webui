import os

os.system("pip install 'mmengine>=0.6.0'")
os.system("pip install 'mmcv>=2.0.0rc4,<2.1.0'")
os.system("pip install mmsegmentation")

import gradio as gr
import fnmatch
import cv2
import numpy as np
import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.apis import MMSegInferencer

import PIL
from mim import download
import warnings

warnings.filterwarnings("ignore")

mmseg_models_list = MMSegInferencer.list_models('mmseg')

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


def download_cfg_checkpoint_model_name(model_name):
    clear_folder("./checkpoint")
    download(package='mmsegmentation',
             configs=[model_name],
             dest_root='./checkpoint')



# 定义推理函数
def predict(img, model_name):
    # 保存输入图片
    img_path = 'input_image.png'
    save_image(img, img_path)
    download_cfg_checkpoint_model_name(model_name)

    config_path = [f for f in os.listdir(path) if fnmatch.fnmatch(f, "*.py")][0]
    config_path = path + "/" + config_path

    checkpoint_path = [f for f in os.listdir(path) if fnmatch.fnmatch(f, "*.pth")][0]
    checkpoint_path = path + "/" + checkpoint_path

    # 从配置文件和权重文件构建模型
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    model = init_model(config_path, checkpoint_path, device=device)

    # 推理给定图像
    result = inference_model(model, img_path)

    # 保存可视化结果
    vis_image = show_result_pyplot(model, img_path, result, show=False)
    vis_image_path = 'output_image.png'
    cv2.imwrite(vis_image_path, vis_image)
    output_img = PIL.Image.open(vis_image_path)
    # 返回输出图片
    return output_img

download_test_image()
# 定义输入和输出界面
inputs_img = gr.inputs.Image(type='pil', label="Input Image")
model_list = gr.inputs.Dropdown(choices=[m for m in mmseg_models_list], label='Model')
outputs_img = gr.outputs.Image(type='pil', label="Output Image")

# 启动界面
title = "MMSegmentation segmentation web demo"
description = "<div align='center'><img src='https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/resources/mmseg-logo.png' width='450''/><div>" \
              "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmsegmentation'>MMSegmentation</a> MMSegmentation 是一个基于 PyTorch 的语义分割开源工具箱。它是 OpenMMLab 项目的一部分。" \
              "OpenMMLab Semantic Segmentation Toolbox and Benchmark..</p>"
article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmsegmentation'>MMSegmentation</a></p>" \
          "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"
examples = [["bus.jpg", "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"],
            ["dogs.jpg", "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"],
            ["zidane.jpg", "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"]
            ]
gr.Interface(fn=predict, inputs=[inputs_img, model_list], outputs=outputs_img, examples=examples,
             title=title, description=description, article=article).launch()
