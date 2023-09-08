#
# from mmpretrain import ImageClassificationInferencer
# image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
# inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
# # Note that the inferencer output is a list of result even if the input is a single sample.
# result = inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')[0]
# pred_class = result['pred_class']
# pred_score = result['pred_score']
# pred_lable = result['pred_label']

import gradio as gr
import cv2
import mmpretrain
import numpy as np
import torch
from mmpretrain import ImageClassificationInferencer
import requests
from io import BytesIO

mmpretrain_model_list = mmpretrain.list_models()


# inferencer = ImageClassificationInferencer(str(model_name))

def draw_text(img, point, text, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.4
    thickness = 5
    text_thickness = 1
    bg_color = (255, 255, 255)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (0, 0, 255), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(img, '%d' % (text), point, fontFace, 0.5, (255, 0, 0))
    return img


def draw_text_line(img, point, text_line: str, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.4
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, drawType)
    return img

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

def predict_with_visualization(img, model_name):
    '''
    :param model:
    :param img:
    :return:
    '''
    # 加载模型
    inferencer = ImageClassificationInferencer(model_name)
    # 读取图片
    img = np.array(img)
    # 进行预测
    result = inferencer(img)[0]
    pred_class = result['pred_class']
    pred_score = result['pred_score']
    pred_label = result['pred_label']

    # 在图片上绘制标签、分数和类别
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Class: {pred_class}\nScore: {pred_score:.2f}\nLabel: {pred_label}"
    text_x = 10
    text_y = 10
    img = draw_text_line(img, (text_x, text_y), text, drawType="custom")
    # 调整图片大小
    max_size = 800
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(img, (int(max_size * img.shape[1] / img.shape[0]), max_size))
    else:
        img = cv2.resize(img, (max_size, int(max_size * img.shape[0] / img.shape[1])))

    # 返回带有标签、分数和类别的图片
    return img, pred_class, pred_score, pred_label


# 创建 gradio 接口
download_test_image()
examples = [
        ['bus.jpg', 'resnet50_8xb32_in1k-300e_coco'],
        ['dogs.jpg', 'swin-tiny_16xb64_in1k'],
        ['zidane.jpg', 'tinyvit-5m_3rdparty_in1k']
]
input_img = gr.inputs.Image(type="pil", label="input")
model_list = gr.inputs.Dropdown(choices=[m for m in mmpretrain_model_list], label='Model',
                                default='resnet50_8xb32_in1k')
output_img = gr.inputs.Image(type="pil", label="output")
output_class = gr.outputs.Textbox(label="class")
output_score = gr.outputs.Textbox(label="score")
output_label = gr.outputs.Textbox(label="label")

iface = gr.Interface(fn=predict_with_visualization,
                     inputs=[input_img, model_list],
                     outputs=[output_img, output_class, output_score, output_label],
                     examples=examples,
                     title="Image Classification with Visualization",
                     description="")
iface.launch()
