import torch
import timm
import gradio as gr
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

# 加载预训练模型
model = timm.create_model('resnet34', pretrained=True, num_classes=1000)
model.eval()

# 定义图像预处理函数
def preprocess_image(image: Image.Image, input_size: int = 224) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# 定义推理函数
def predict(image: Image.Image) -> Image.Image:
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, class_idx = probabilities.max(dim=1)

    # 将预测结果转换为类名
    # imagenet_labels = ImageNet.classes
    # predicted_class = imagenet_labels[class_idx.item()]
    predicted_class =  class_idx.item()

    # 在输入图像上添加预测结果
    image_with_prediction = image.copy()
    draw = ImageDraw.Draw(image_with_prediction)
    draw.text((10, 10), predicted_class, fill=(255, 0, 0))

    return image_with_prediction

# 使用gradio创建交互界面
image_input = gr.inputs.Image(type='pil')
image_output = gr.outputs.Image(type='pil')

gr.Interface(fn=predict, inputs=image_input, outputs=image_output).launch()