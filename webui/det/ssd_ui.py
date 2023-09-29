
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