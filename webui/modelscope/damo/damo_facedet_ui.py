import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage

import warnings
warnings.filterwarnings("ignore")

# 定义模型
face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd')


# 定义推理函数
def detect_faces(img_pil):
    img_dir = "input_img.jpg"
    img_pil.save(img_dir)
    # 进行人脸检测
    result = face_detection(img_dir)
    # 可视化结果
    img_cv = draw_face_detection_result(img_dir, result)
    # 将结果转换为 Gradio 的输出格式
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_out_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_out_pil


# 定义输入和输出
inputs = gr.inputs.Image(type='pil', label="input")
outputs = gr.outputs.Image(type='pil', label="output")

# 启动 Gradio 应用
gr.Interface(fn=detect_faces, inputs=inputs, outputs=outputs, title="人脸检测",
             description="使用 damo/cv_ddsar_face-detection_iclr23-damofd 模型进行人脸检测。").launch()
