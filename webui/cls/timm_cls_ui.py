import gradio as gr
import timm
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# 获取 ImageNet 类别列表
imagenet_classes = []
with open("../../data/imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]


# 预处理
def preprocess(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# 后处理
def postprocess(output: torch.Tensor) -> (str, float):
    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
    _, index = output.max(1)
    label = imagenet_classes[index.item()]
    confidence = probabilities[index].item()
    return label, confidence


def predict(image: Image.Image, model_name: str) -> (str, float):
    model = get_model(model_name)
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
    return postprocess(output)


def get_model(model_name: str):
    try:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
    except ValueError:
        raise ValueError(f"Model {model_name} not found or pretrained weights not available.")
    return model


# 将分类名称和置信度绘制在图像上
def draw_label_on_image(image: Image.Image, label: str, confidence: float) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{label} ({confidence:.2f}%)"
    draw.text((10, 10), text, font=font, fill=(255, 255, 255, 128))
    return image


def predict_and_draw(image: Image.Image, model_name: str) -> Image.Image:
    label, confidence = predict(image, model_name)
    image_output = draw_label_on_image(image, label, confidence)
    return image_output, label, confidence


# 定义输入和输出
image_input = gr.inputs.Image(type="pil")
model_input = gr.inputs.Dropdown(choices=["resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                                          "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "mobilenetv3_large_100",
                                          "mobilenetv3_rw", "resnet18", "resnet34", "resnet101", "resnext50_32x4d", "resnext101_32x8d",
                                          "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"], label="Model",default="resnet50")
image_output = gr.outputs.Image(type="pil")
text_output = gr.outputs.Textbox(label="Label", type="text")
score_output = gr.outputs.Textbox(label="Confidence", type="text")

# 创建 Gradio 界面
iface = gr.Interface(fn=predict_and_draw, inputs=[image_input, model_input], outputs=[image_output, text_output, score_output],
                     title="Image Classification on timm", description="Upload an image and classify，it.")

# 启动 Gradio 界面
iface.launch()