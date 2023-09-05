import gradio as gr
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ImageClassifier:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.load_classes()

    def load_model(self, model_name):
        if model_name == "ResNet18":
            model = models.resnet18(pretrained=True)
        elif model_name == "AlexNet":
            model = models.alexnet(pretrained=True)
        elif model_name == "VGG16":
            model = models.vgg16(pretrained=True)
        elif model_name == "GoogLeNet":
            model = models.googlenet(pretrained=True)
        elif model_name == "ResNet50":
            model = models.resnet50(pretrained=True)
        elif model_name == "DenseNet121":
            model = models.densenet121(pretrained=True)
        elif model_name == "MobileNetV2":
            model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError("Invalid model name")
        return model

    def load_classes(self):
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def classify_image(self, image):
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
        # 使用模型进行预测
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            score = torch.nn.functional.softmax(output, dim=1)[0][predicted[0]].item()

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # 在图片上绘制预测结果
        label = f"{self.classes[predicted[0]]} ({score:.2f})"
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        return image, self.classes[predicted[0]]

    def classify_images(self, images):
        results = []
        for image in images:
            result = self.classify_image(image)
            results.append(result)
        return results

    def classify_interactive(self):
        input_image = gr.inputs.Image(type='pil')
        output_image = gr.outputs.Image(type='pil')
        output_text = gr.outputs.Textbox()
        model_name = gr.inputs.Dropdown(["ResNet18", "AlexNet", "VGG16", "GoogLeNet", "ResNet50", "DenseNet121", "MobileNetV2"], label="Model", default="VGG16")

        def predict(image, model_name):
            self.model = self.load_model(model_name)
            result_image, result_class = self.classify_image(image)
            return result_image, result_class

        gr.Interface(fn=predict, inputs=[input_image, model_name], outputs=[output_image, output_text],
                     title="Image Classification", description="Upload an image and classify it.").launch()


if __name__ == "__main__":
    classifier = ImageClassifier("VGG16")
    classifier.classify_interactive()