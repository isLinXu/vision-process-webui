import torch
import cv2
from PIL import Image
import numpy as np
import gradio as gr
import warnings
import wget

warnings.filterwarnings("ignore")

class YOLOv3WebUI:
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

    def download_weights(self, url):
        filename = wget.download(url)
        print(filename)

    def detect_objects(self, img, conf, iou, line_width, device, model_type, model_path):

        # choose model type
        if model_type == "yolov3":
            url = 'https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3.pt'
            self.download_weights(url)
            self.model = torch.hub.load('ultralytics/yolov3', 'custom', path='yolov3.pt', device=device)
        elif model_type == "yolov3-tiny":
            url = 'https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3-tiny.pt'
            self.download_weights(url)
            self.model = torch.hub.load('ultralytics/yolov3', 'custom', path='yolov3-tiny.pt', device=device)
        elif model_type == "yolov3-spp":
            url = 'https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-spp.pt'
            self.download_weights(url)
            self.model = torch.hub.load('ultralytics/yolov3', 'custom', path='yolov3-spp.pt', device=device)
        else:
            self.model = torch.hub.load('ultralytics/yolov3', 'custom', path=model_path, device=device)
        if model_type not in ["yolov3", "yolov3-tiny", "yolov3-spp"]:
            self.model = torch.hub.load('ultralytics/yolov3', 'custom', path=model_path, device=device)

        # Convert input image to numpy array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert numpy array to PIL image
        img = Image.fromarray(img)
        # Use YOLOv3 model for inference
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
    # Instantiate YOLOv3WebUI class
    detector = YOLOv3WebUI()
    examples = [
        ['bus.jpg', 0.25, 0.45, 2, "cpu", "yolov3", "yolov3.pt"],
        ['dogs.jpg', 0.25, 0.45, 2, "cpu", "yolov3", "yolov3.pt"],
        ['zidane.jpg', 0.25, 0.45, 2, "cpu", "yolov3", "yolov3.pt"]
    ]
    # Define Gradio interface
    iface = gr.Interface(
        fn=detector.detect_objects,
        # inputs="image",
        inputs=["image",
                gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.25,
                                 label="Confidence Threshold"),
                gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.45,
                                 label="IoU Threshold"),
                gr.inputs.Number(default=2, label="Line Width"),
                gr.inputs.Radio(["cpu", "cuda"], label="Device", default="cpu"),
                gr.inputs.Radio(["yolov3", "yolov3-tiny", "yolov3-spp"], label="Model Type", default="yolov3"),
                gr.inputs.Textbox(default="yolov3.pt", label="Model Path")],
        outputs="image",
        examples=examples,
        title="YOLOv3 Object Detector",
        description="Detect objects in an image using YOLOv3 model.",
        theme="default",
        layout="vertical",
        allow_flagging=False,
        analytics_enabled=True,
        server_port=None,
        server_name=None,
        server_protocol=None,
    )

    # Run Gradio interface
    iface.launch()
