import gradio as gr
import torch
import cv2
from PIL import Image
import numpy as np

class YOLOv5Detector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='../weights/yolov5/yolov5s.pt')
        self.model.conf = 0.25

    def detect_objects(self, img):
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
            # Draw bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            # Draw label name
            label = f"{name}: {confidence:.2f}"
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

if __name__ == '__main__':
    # Instantiate YOLOv5Detector class
    detector = YOLOv5Detector()

    # Define Gradio interface
    iface = gr.Interface(
        fn=detector.detect_objects,
        inputs="image",
        outputs="image",
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
    iface.launch()