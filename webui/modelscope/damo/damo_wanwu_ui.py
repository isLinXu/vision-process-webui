import gradio as gr
import cv2
import PIL.Image as Image
import numpy as np
import torch
from PIL import ImageFont
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import warnings
warnings.filterwarnings("ignore")


def recognize_image(img):
    # 加载模型
    general_recognition = pipeline(Tasks.general_recognition, model='damo/cv_resnest101_general_recognition')
    img = np.array(img)
    # 模型推理
    result = general_recognition(img)
    print(result)
    # 绘制分数和类别信息
    scores = result['scores']
    labels = result['labels']

    # 读取中文字体文件
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    thickness = 2

    for i in range(len(scores)):
        score = scores[i]
        label = labels[i]
        img = cv2.putText(img, f"{label}: {score:.2f}", (10, 30), font, font_scale, font_color, thickness)

    img_pil = Image.fromarray(np.uint8(img))
    return img_pil, scores, labels


def download_test_image():
    # Images
    torch.hub.download_url_to_file(
        'https://pailitao-image-recog.oss-cn-zhangjiakou.aliyuncs.com/mufan/img_data/maas_test_data/dog.png',
        'dog.png')


download_test_image()
input_image = gr.inputs.Image(type='pil', label="input image")
output_image = gr.outputs.Image(type='pil', label="output image")
out_scores = gr.outputs.Textbox(label="scores")
out_lables = gr.outputs.Textbox(label="labels")
examples = [["dog.png"]]
title = "万物识别-中文-通用领域 web demo"
interface = gr.Interface(fn=recognize_image,
                         inputs=input_image,
                         outputs=[output_image, out_scores, out_lables],
                         examples=examples,
                         title=title)

interface.launch()
