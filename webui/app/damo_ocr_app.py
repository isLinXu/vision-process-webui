'''
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

### 使用url
img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
result = ocr_recognition(img_url)
print(result)
'''
import os

os.system("pip install opencv-python")
os.system("pip install tensorflow")
os.system("pip install modelscope")

import gradio as gr
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import warnings

warnings.filterwarnings("ignore")

def ocr_recognition(img_url):
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
    result = ocr_recognition(img_url)
    return result


def download_test_image():
    # Images
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/assets/59380685/6cdbe53c-eb34-4310-8bd4-18b0a1aff803',
        'ocr_test.jpg')


download_test_image()
input_image = gr.inputs.Image()
output_text = gr.outputs.Textbox()

title = "DAMO-OCR web demo"
description = "<div align='center'><img src='https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/assets/logo.png' width='800''/><div>" \
              "<p style='text-align: center'><a href='https://github.com/tinyvision/DAMO-YOLO'>DAMO-YOLO</a> DAMO-YOLO：一种快速准确的目标检测方法，采用了一些新技术，包括 NAS 主干、高效的 RepGFPN、ZeroHead、AlignedOTA 和蒸馏增强。" \
              "DAMO-OCR: a fast and accurate object detection method with some new techs, including NAS backbones, efficient RepGFPN, ZeroHead, AlignedOTA, and distillation enhancement..</p>"
article = "<p style='text-align: center'><a href='https://github.com/tinyvision/DAMO-OCR'>DAMO-YOLO</a></p>" \
          "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

examples = [["ocr_test.jpg"]]

gr.Interface(fn=ocr_recognition, inputs=input_image,
             outputs=output_text, examples=examples,
             title="OCR Recognition").launch()
