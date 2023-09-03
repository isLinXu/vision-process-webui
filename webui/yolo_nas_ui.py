import cv2
import numpy as np
from super_gradients.training import models
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
yolo_nas_s = models.get("yolo_nas_s", pretrained_weights="coco").to(device)

def predict(image_path):
    results = yolo_nas_s.predict(image_path)

    # 获取图片数据和bbox信息
    image = results._images_prediction_lst[0].image
    bboxes_xyxy = results._images_prediction_lst[0].prediction.bboxes_xyxy
    labels = results._images_prediction_lst[0].prediction.labels
    class_names = results._images_prediction_lst[0].class_names

    # 绘制矩形框和标签名称
    for bbox, label in zip(bboxes_xyxy, labels):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_name = class_names[int(label)]
        cv2.putText(image, label_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 返回绘制好矩形框和标签名称的图片
    return image

import gradio as gr

iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(),
    outputs="image"
)

iface.launch()