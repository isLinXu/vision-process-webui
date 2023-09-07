
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

import warnings

warnings.filterwarnings("ignore")

# 加载模型
models_dict = {
    'DeepLabv3': models.segmentation.deeplabv3_resnet50(pretrained=True).eval(),
    'DeepLabv3+': models.segmentation.deeplabv3_resnet101(pretrained=True).eval(),
    'FCN-ResNet50': models.segmentation.fcn_resnet50(pretrained=True).eval(),
    'FCN-ResNet101': models.segmentation.fcn_resnet101(pretrained=True).eval(),
    'LRR': models.segmentation.lraspp_mobilenet_v3_large(pretrained=True).eval(),
}

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

def download_test_img():
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

def predict_segmentation(image, model_name):
    download_test_img()
    # 图像预处理
    image_tensor = image_transforms(image).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        output = models_dict[model_name](image_tensor)['out'][0]
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


import gradio as gr

examples = [
    ['bus.jpg', 'DeepLabv3'],
    ['dogs.jpg', 'DeepLabv3'],
    ['zidane.jpg', 'DeepLabv3']
]

model_list = ['DeepLabv3', 'DeepLabv3+', 'FCN-ResNet50', 'FCN-ResNet101', 'LRR']
inputs = [
    gr.inputs.Image(type='pil', label='原始图像'),
    gr.inputs.Dropdown(model_list, label='选择模型', default='DeepLabv3')
]
outputs = [
    gr.outputs.Image(type='pil',label='分割图'),
    gr.outputs.Image(type='pil',label='融合图')
]
interface = gr.Interface(
    predict_segmentation,
    inputs,
    outputs,
    examples=examples,
    capture_session=True,
    title='torchvision-segmentation-webui',
    description='torchvision segmentation webui on gradio'
)

interface.launch()