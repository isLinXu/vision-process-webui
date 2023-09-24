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
os.system("pip install tensorflow")
os.system("pip install modelscope")


import gradio as gr
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

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
examples = [["ocr_test.jpg"]]

gr.Interface(fn=ocr_recognition, inputs=input_image,
             outputs=output_text, examples=examples,
             title="OCR Recognition").launch()