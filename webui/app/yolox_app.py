
import cv2
import numpy as np
import gradio as gr
import torch

from super_gradients.training import models
import warnings

warnings.filterwarnings("ignore")
class YOLOX_WebUI:
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

    def predict(self, image_path,conf, iou, line_width, device, model_type, model_path):
        self.device = device
        if model_type == "yolox_n":
            self.model = models.get("yolox_n", pretrained_weights="coco").to(self.device)
        elif model_type == "yolox_s":
            self.model = models.get("yolox_s", pretrained_weights="coco").to(self.device)
        elif model_type == "yolox_m":
            self.model = models.get("yolox_m", pretrained_weights="coco").to(self.device)
        elif model_type == "yolox_t":
            self.model = models.get("yolox_t", pretrained_weights="coco").to(self.device)
        elif model_type == "yolox_l":
            self.model = models.get("yolox_l", pretrained_weights="coco").to(self.device)
        else:
            self.model = models.get(model_path, pretrained_weights="coco").to(self.device)
        if model_type not in ["yolox_n", "yolox_s", "yolox_m", "yolox_l","yolox_t"]:
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
    detector = YOLOX_WebUI()
    examples = [
        ['bus.jpg', 0.25, 0.45, 2, "cpu", "yolox_n", "yolox_n.pt"],
        ['dogs.jpg', 0.25, 0.45, 2, "cpu", "yolox_n", "yolox_n.pt"],
        ['zidane.jpg', 0.25, 0.45, 2, "cpu", "yolox_n", "yolox_n.pt"]
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
                gr.inputs.Radio(["yolox_n", "yolox_s", "yolox_m", "yolox_l","yolox_t"],
                                label="Model Type", default="yolo_nas_s"),
                gr.inputs.Textbox(default="yolox_n.pt", label="Model Path")],
        outputs="image",
        examples=examples,
        title="YOLOX_WebUI Object Detector",
        description="Detect objects in an image using YOLOX model.",
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


