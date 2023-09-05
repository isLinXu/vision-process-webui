import torch
import torchvision
import gradio as gr
from PIL import Image, ImageDraw
import warnings

warnings.filterwarnings("ignore")

# 定义模型列表
model_list = [
    {
        'name': 'fasterrcnn_resnet50_fpn',
        'model': torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'pretrained': True,
        'score_threshold': 0.5
    },
    {
        'name': 'fasterrcnn_mobilenet_v3_large_320_fpn',
        'model': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        'pretrained': True,
        'score_threshold': 0.5
    },
    {
        'name': 'maskrcnn_resnet50_fpn',
        'model': torchvision.models.detection.maskrcnn_resnet50_fpn,
        'pretrained': True,
        'score_threshold': 0.5
    },
    {
        'name': 'retinanet_resnet50_fpn',
        'model': torchvision.models.detection.retinanet_resnet50_fpn,
        'pretrained': True,
        'score_threshold': 0.5
    },
    {
        'name': 'keypointrcnn_resnet50_fpn',
        'model': torchvision.models.detection.keypointrcnn_resnet50_fpn,
        'pretrained': True,
        'score_threshold': 0.5
    },
    {
        'name': 'fasterrcnn_mobilenet_v3_large_fpn',
        'model': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        'pretrained': True,
        'score_threshold': 0.5
    },
]

# 定义类别名称
class_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# 预处理函数
def preprocess(image):
    # 将图像转换为 PyTorch 张量
    image_tensor = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)
    # 归一化图像
    image_tensor /= 255.0
    return image_tensor.unsqueeze(0)


# 后处理函数
def postprocess(output, score_threshold):
    # 获取模型输出中的框、分数和类别
    boxes = output[0]['boxes'].detach().numpy()
    scores = output[0]['scores'].detach().numpy()
    classes = output[0]['labels'].detach().numpy()
    # 将结果拼接为列表
    results = []
    for box, score, cls in zip(boxes, scores, classes):
        if score >= score_threshold:
            results.append({
                'box': box.tolist(),
                'score': score.tolist(),
                'class': cls.tolist(),
                'label': class_names[cls]
            })
    return results


# 定义推理函数
def predict(image, model_name, score_threshold):
    # 获取模型参数
    model_info = next((m for m in model_list if m['name'] == model_name), None)
    if model_info is None:
        raise ValueError('Invalid model name')
    model_fn = model_info['model']
    pretrained = model_info['pretrained']
    # 加载模型
    model = model_fn(pretrained=pretrained)
    model.eval()
    # 预处理图像
    image_tensor = preprocess(image)
    # 运行模型
    output = model(image_tensor)
    # 后处理结果
    results = postprocess(output, score_threshold)
    # 绘制检测结果
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for result in results:
        box = result['box']
        label = result['label'] + ':' + str(result['score'])
        draw.rectangle(box, outline='blue', width=3)
        draw.text((box[0], box[1]), label, fill='blue', width=3)
    # 返回检测结果和效果图
    return img, results


iface = gr.Interface(fn=predict,
                     inputs=[gr.inputs.Image(),
                             gr.inputs.Dropdown(choices=[m['name'] for m in model_list], label='Model'),
                             gr.inputs.Slider(minimum=0, maximum=1, step=0.05, default=0.5, label='Score Threshold')],
                     outputs=[gr.outputs.Image(type='pil'), gr.outputs.JSON()],
                     title='Torchvision-detection-webui',
                     description='Torchvision-detection-webui on gradio')

iface.launch()
