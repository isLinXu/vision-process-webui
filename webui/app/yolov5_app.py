import os

import requests
import torch
import cv2
from PIL import Image
import numpy as np
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

class YOLOv5WebUI:
    def __init__(self):
        self.download_test_img()

    def download_test_img(self):
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

    def detect_objects(self, img, conf, iou, line_width, device, model_type, model_path):
        # choose model type
        if model_type == "yolov5n":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device=device)
        elif model_type == "yolov5s":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
        elif model_type == "yolov5m":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', device=device)
        elif model_type == "yolov5l":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', device=device)
        elif model_type == "yolov5x":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device=device)

        if model_type not in ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)

        # Convert input image to numpy array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert numpy array to PIL image
        img = Image.fromarray(img)
        # Use YOLOv5 model for inference
        results = self.model(img)
        xyxy = results.pandas().xyxy[0]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Iterate through each bbox, draw bounding box and label name
        for bbox in xyxy.itertuples():
            # Generate random color
            color = tuple(np.random.randint(0, 255, 3).tolist())
            xmin, ymin, xmax, ymax = bbox[1:5]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = bbox[5]
            name = bbox[7]
            if confidence > conf:
                # Draw bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                # Draw label name
                label = f"{name}: {confidence:.2f}"
                cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img


if __name__ == '__main__':
    # Instantiate YOLOv5WebUI class
    detector = YOLOv5WebUI()

    examples = [
        ['bus.jpg', 0.25, 0.45, 2, "cpu", "yolov5n", "yolov5s.pt"],
        ['dogs.jpg', 0.25, 0.45, 2, "cpu", "yolov5s", "yolov5s.pt"],
        ['zidane.jpg', 0.25, 0.45, 2, "cpu", "yolov5m", "yolov5s.pt"]
    ]

    # Define Gradio interface
    iface = gr.Interface(
        fn=detector.detect_objects,
        inputs=["image",
                gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.25,
                                 label="Confidence Threshold"),
                gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.45,
                                 label="IoU Threshold"),
                gr.inputs.Number(default=2, label="Line Width"),
                gr.inputs.Radio(["cpu", "cuda"], label="Device", default="cpu"),
                gr.inputs.Radio(["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                                label="Model Type", default="yolov5n"),
                gr.inputs.Textbox(default="yolov5s.pt", label="Model Path")],
        outputs="image",
        examples=examples,
        title="YOLOv5 Object Detector",
        description="Detect objects in an image using YOLOv5 model.",
        theme="default",
        layout="vertical",
        allow_flagging=False,
        analytics_enabled=True,
        server_port=None,
        server_name=None,
        server_protocol=None,
    )

    # Run Gradio interface
    iface.launch(share=True)
