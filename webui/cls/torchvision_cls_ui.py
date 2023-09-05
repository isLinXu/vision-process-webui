import gradio as gr
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def classify_image(model, image):
    # 对输入图片进行预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image_tensor = transform(image).unsqueeze(0)
    with open('../../data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    # 使用模型进行预测
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        score = torch.nn.functional.softmax(output, dim=1)[0][predicted[0]].item()

    print("predicted:", predicted, "score:", score)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 在图片上绘制预测结果
    label = f"{classes[predicted[0]]} ({score:.2f})"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image


resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {"ResNet18": resnet18, "AlexNet": alexnet, "VGG16": vgg16}


def predict(model_name, image):
    model = models[model_name]
    image = np.array(image)
    image = Image.fromarray(image.astype('uint8'))
    return classify_image(model, image)


input_image = gr.inputs.Image(type='pil')
output_image = gr.outputs.Image(type='pil')

gr.Interface(fn=predict, inputs=[gr.inputs.Dropdown(list(models.keys()), label="Model"), input_image],
             outputs=output_image, title="Image Classification",
             description="Select a model and upload an image.").launch()
