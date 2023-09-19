
import gradio as gr
import PIL.Image as Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import numpy as np
import random

# 加载模型
p = pipeline(task='image-object-detection', model='damo/cv_tinynas_object-detection_damoyolo', device='cpu')

def object_detection(img_pil, confidence_threshold):
    # 传入图片进行推理
    result = p(img_pil)
    # 读取图片
    img_cv = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    # 获取bbox和类别
    scores = result['scores']
    boxes = result['boxes']
    labels = result['labels']
    # 遍历每个bbox
    for i in range(len(scores)):
        # 只绘制置信度大于设定阈值的bbox
        if scores[i] > confidence_threshold:
            # 随机生成颜色
            class_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # 获取bbox坐标
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 绘制bbox
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), class_color, thickness=2)
            # 绘制类别标签
            label = f"{labels[i]}: {scores[i]:.2f}"
            cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, thickness=2)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

# 定义输入和输出
input_image = gr.inputs.Image(type='pil')
output_image = gr.outputs.Image(type='pil')

# 创建 Gradio 接口并运行
gr.Interface(
    fn=object_detection,
    inputs=[
        input_image,
        gr.inputs.Slider(minimum=0, maximum=1, step=0.05, default=0.5, label="Confidence Threshold")
    ],
    outputs=output_image,
    title='Object Detection'
).launch()