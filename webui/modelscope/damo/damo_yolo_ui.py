import gradio as gr
import PIL.Image as Image
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import numpy as np
import random

import warnings

warnings.filterwarnings("ignore")


def object_detection(img_pil, model_name, confidence_threshold, device):
    # 加载模型
    p = pipeline(task='image-object-detection', model=model_name, device=device)

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
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, thickness=2)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil


damo_model_list = [
    'damo/cv_cspnet_image-object-detection_yolox',
    'damo/cv_cspnet_image-object-detection_yolox_nano_coco',
    'damo/cv_swinl_image-object-detection_dino',
    'damo/cv_tinynas_object-detection_damoyolo',
    'damo/cv_tinynas_object-detection_damoyolo-m',
    'damo/cv_tinynas_object-detection_damoyolo-t',
    'damo/cv_vit_object-detection_coco',
    'damo/cv_tinynas_detection',
]


def download_test_image():
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266264420-21575a83-4057-41cf-8a4a-b3ea6f332d79.jpg',
        '../bus.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266264536-82afdf58-6b9a-4568-b9df-551ee72cb6d9.jpg',
        '../dogs.jpg')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/266264600-9d0c26ca-8ba6-45f2-b53b-4dc98460c43e.jpg',
        '../zidane.jpg')


if __name__ == '__main__':
    download_test_image()
    # 定义输入和输出
    input_image = gr.inputs.Image(type='pil')
    model_name = gr.inputs.Dropdown(choices=[m for m in damo_model_list], label='Model',
                                    default='damo/cv_tinynas_object-detection_damoyolo')
    input_slide = gr.inputs.Slider(minimum=0, maximum=1, step=0.05, default=0.5, label="Confidence Threshold")
    input_device = gr.inputs.Radio(["cpu", "cuda", "gpu"], default="cpu")
    output_image = gr.outputs.Image(type='pil')

    examples = [['bus.jpg', "damo/cv_tinynas_object-detection_damoyolo", 0.45, "cpu"],
                ['dogs.jpg', "damo/cv_tinynas_object-detection_damoyolo", 0.45, "cpu"],
                ['zidane.jpg', "damo/cv_tinynas_object-detection_damoyolo", 0.45, "cpu", ]]
    title = "DAMO-YOLO web demo"
    description = "<div align='center'><img src='https://raw.githubusercontent.com/tinyvision/DAMO-YOLO/master/assets/logo.png' width='800''/><div>" \
                  "<p style='text-align: center'><a href='https://github.com/tinyvision/DAMO-YOLO'>DAMO-YOLO</a> DAMO-YOLO DAMO-YOLO DAMO-YOLO：一种快速准确的目标检测方法，采用了一些新技术，包括 NAS 主干、高效的 RepGFPN、ZeroHead、AlignedOTA 和蒸馏增强。" \
                  "DAMO-YOLO: a fast and accurate object detection method with some new techs, including NAS backbones, efficient RepGFPN, ZeroHead, AlignedOTA, and distillation enhancement..</p>"
    article = "<p style='text-align: center'><a href='https://github.com/tinyvision/DAMO-YOLO'>DAMO-YOLO</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

    # 创建 Gradio 接口并运行
    gr.Interface(
        fn=object_detection,
        inputs=[
            input_image, model_name, input_slide, input_device
        ],
        outputs=output_image,
        title=title,
        examples=examples,
        description=description, article=article
    ).launch()
