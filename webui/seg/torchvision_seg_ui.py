import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr

# 加载模型
model = models.segmentation.fcn_resnet101(pretrained=True).eval()

# 图像预处理
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 定义推理函数
def predict_segmentation(image):
    # 图像预处理
    image_tensor = image_transforms(image).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
        output_predictions = output.argmax(0)
        segmentation = F.interpolate(
            output.float().unsqueeze(0),
            size=image.size[::-1],
            mode='bicubic',
            align_corners=False
        )[0].argmax(0).numpy()

    # 分割图
    segmentation_image = np.uint8(segmentation)
    segmentation_image = cv2.applyColorMap(segmentation_image, cv2.COLORMAP_JET)

    # 融合图
    blend_image = cv2.addWeighted(np.array(image), 0.5, segmentation_image, 0.5, 0)
    blend_image = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB)

    return segmentation_image, blend_image

# Gradio 接口
inputs = gr.inputs.Image(type='pil', label='原始图像')
outputs = [gr.outputs.Image(type='pil',label='分割图'),
           gr.outputs.Image(type='pil',label='融合图')]
interface = gr.Interface(
    predict_segmentation,
    inputs,
    outputs,
    capture_session=True,
    title='torchvision-segmentation-webui',
    description='torchvision segmentation webui on gradio'
)

# 启动 Gradio 接口
interface.launch()