# import torch
# import gradio as gr
# from torchvision import transforms
# from PIL import Image
# import cv2
#
# # 加载 SSD 模型
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
#
# # 加载本地权重
# checkpoint = torch.load('ssd300_mAP_77.43_v2.pth', map_location='cpu')
# model.load_state_dict(checkpoint['model'])
#
# # 设置模型为评估模式
# model.eval()
#
# # 定义类别标签
# CLASSES = [
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]
#
# # 定义图像预处理函数
# def preprocess(image):
#     transform = transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)
#
# # 定义推理函数
# def predict(image):
#     # 预处理图像
#     image = preprocess(image)
#     # 获取模型输出
#     outputs = model(image)
#     # 获取检测结果
#     results = model.decode_results(outputs)
#     # 获取检测框、类别和置信度
#     boxes, classes, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
#     # 在图像上绘制检测框和类别标签
#     image = image.squeeze().permute(1, 2, 0).numpy()
#     for box, cls, score in zip(boxes, classes, scores):
#         if score > 0.5:
#             x1, y1, x2, y2 = box.int().tolist()
#             label = CLASSES[cls]
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             image = cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     # 返回检测后的图像
#     return (image * 255).astype('uint8')
#
# # 创建 Gradio 接口
# input_image = gr.inputs.Image(label='输入图片')
# output_image = gr.outputs.Image(label='检测后的推理图片')
# interface = gr.Interface(fn=predict, inputs=input_image, outputs=output_image, title='SSD 目标检测 Demo')
# interface.launch()

# import torch
# import gradio as gr
# from torchvision import transforms
# import cv2
# from PIL import Image
#
# # 加载 SSD 模型
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
#
# # 加载本地权重
# checkpoint = torch.load('ssd300_mAP_77.43_v2.pth', map_location='cpu')
# model.load_state_dict(checkpoint['model'])
#
# # 设置模型为评估模式
# model.eval()
#
# # 定义类别标签
# CLASSES = [
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]
#
# # 定义图像预处理函数
# def preprocess(image):
#     transform = transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)
#
# # 定义推理函数
# def predict(image):
#     # 预处理图像
#     image = preprocess(image)
#     # 获取模型输出
#     outputs = model(image)
#     # 获取检测结果
#     results = model.decode_results(outputs)
#     # 获取检测框、类别和置信度
#     boxes, classes, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
#     # 在图像上绘制检测框和类别标签
#     image = image.squeeze().permute(1, 2, 0).numpy()
#     for box, cls, score in zip(boxes, classes, scores):
#         if score > 0.5:
#             x1, y1, x2, y2 = box.int().tolist()
#             label = CLASSES[cls]
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             image = cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     # 返回检测后的图像
#     return (image * 255).astype('uint8')
#
# # 创建 Gradio 接口
# input_image = gr.inputs.Image(label='输入图片')
# output_image = gr.outputs.Image(label='检测后的推理图片')
# interface = gr.Interface(fn=predict, inputs=input_image, outputs=output_image, title='SSD 目标检测 Demo')
# interface.launch()

import torch
import cv2
import gradio as gr

# 加载 ssd 模型
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')

# 加载本地权重
model.load_state_dict(torch.load('nvidia_ssd.pth'), map_location=torch.device('cpu'))

# 设置模型为评估模式
model.eval()

# 类别标签
CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 预处理函数
def preprocess(img):
    # 缩放图片并转换为 RGB 格式
    img = cv2.cvtColor(cv2.resize(img, (300, 300)), cv2.COLOR_BGR2RGB)
    # 转换为 PyTorch Tensor 格式
    img = torch.tensor(img.transpose(2, 0, 1)).float()
    # 归一化
    img /= 255.0
    # 增加 batch 维度
    img = img.unsqueeze(0)
    return img

# 后处理函数
def postprocess(img, predictions):
    # 取出置信度大于 0.6 的预测结果
    boxes = predictions[0]['boxes'][predictions[0]['scores'] > 0.6].detach().cpu().numpy()
    labels = predictions[0]['labels'][predictions[0]['scores'] > 0.6].cpu().numpy()
    scores = predictions[0]['scores'][predictions[0]['scores'] > 0.6].detach().cpu().numpy()

    # 绘制检测框和标签
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{CLASSES[label]} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

# 定义演示函数
def ssd_demo(img):
    # 预处理
    img = preprocess(img)
    # 推理
    predictions = model(img)
    # 后处理
    img = postprocess(img.squeeze(0).permute(1, 2, 0).numpy() * 255.0, predictions)
    return img

# 定义 gradio 接口
iface = gr.Interface(fn=ssd_demo, inputs="image", outputs="image")

# 启动 gradio 接口
iface.launch()