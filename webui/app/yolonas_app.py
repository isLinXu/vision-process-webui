
import torch
import cv2
from PIL import Image
import numpy as np
import gradio as gr

from super_gradients.training import models
import warnings

warnings.filterwarnings("ignore")
class YOLO_NAS_WebUI:
    def __init__(self):
        pass

    def predict(self, image_path,conf, iou, line_width, device, model_type, model_path):
        self.device = device
        if model_type == "yolo_nas_s":
            self.model = models.get("yolo_nas_s", pretrained_weights="coco").to(self.device)
        elif model_type == "yolo_nas_m":
            self.model = models.get("yolo_nas_m", pretrained_weights="coco").to(self.device)
        elif model_type == "yolo_nas_l":
            self.model = models.get("yolo_nas_l", pretrained_weights="coco").to(self.device)
        else:
            self.model = models.get(model_path, pretrained_weights="coco").to(self.device)
        if model_type not in ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]:
            self.model = models.get(model_path, pretrained_weights="coco").to(self.device)

        results = self.model.predict(image_path)

        # get image data and bbox information
        image = results._images_prediction_lst[0].image
        class_names = results._images_prediction_lst[0].class_names
        prediction = results._images_prediction_lst[0].prediction
        bboxes_xyxy = prediction.bboxes_xyxy
        labels = prediction.labels
        confidences = prediction.confidence

        # draw rectangles and label names
        for bbox, label, confidence in zip(bboxes_xyxy, labels, confidences):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x1, y1, x2, y2 = bbox.astype(int)
            if confidence > conf:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cla_name = class_names[int(label)]
                label_name = f"{cla_name}: {confidence:.2f}"
                cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

if __name__ == '__main__':
    # Instantiate YOLO_NAS_WebUI class
    detector = YOLO_NAS_WebUI()
    examples = [
        ['../../images/bus.jpg'],
        ['../../images/dogs.jpg'],
        ['../../images/zidane.jpg']
    ]
    # Define Gradio interface
    iface = gr.Interface(
        fn=detector.predict,
        inputs=["image",
                gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.25,
                                 label="Confidence Threshold"),
                gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.45,
                                 label="IoU Threshold"),
                gr.inputs.Number(default=2, label="Line Width"),
                gr.inputs.Radio(["cpu", "cuda"], label="Device", default="cpu"),
                gr.inputs.Radio(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"],
                                label="Model Type", default="yolo_nas_s"),
                gr.inputs.Textbox(default="yolo_nas_s.pt", label="Model Path")],
        outputs="image",
        examples=examples,
        title="YOLO-NAS Object Detector",
        description="Detect objects in an image using YOLO-NAS model.",
        theme="default",
        layout="vertical",
        allow_flagging=True,
        analytics_enabled=True,
        server_port=None,
        server_name=None,
        server_protocol=None,
    )

    # Run Gradio interface
    iface.launch(share=True)


