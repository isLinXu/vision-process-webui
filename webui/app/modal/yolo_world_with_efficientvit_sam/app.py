# """Fast text to segmentation with yolo-world and efficient-vit sam."""
import os
os.system("pip install 'inference[yolo-world]==0.9.13'")
os.system("pip install supervision==0.18.0")
os.system("pip install timm==0.9.12")
os.system("pip install onnx==1.15.0")
os.system("pip install onnxsim==0.4.35")
os.system("pip install segment-anything")

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from inference.models import YOLOWorld
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model


class ImageSegmenter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_world = self.load_yolo_world()
        self.sam = self.load_sam()
        self.annotators = self.load_annotators()

    def load_yolo_world(self):
        os.system("make model")
        return YOLOWorld(model_id="yolo_world/l")

    def load_sam(self):
        return EfficientViTSamPredictor(
            create_sam_model(name="xl1", weight_url="xl1.pt").to(self.device).eval()
        )

    @staticmethod
    def load_annotators():
        return {
            'bounding_box': sv.BoundingBoxAnnotator(),
            'mask': sv.MaskAnnotator(),
            'label': sv.LabelAnnotator()
        }

    def detect(self, image: np.ndarray, query: str, confidence_threshold: float, nms_threshold: float) -> np.ndarray:
        categories = [category.strip() for category in query.split(",")]
        self.yolo_world.set_classes(categories)

        results = self.yolo_world.infer(image, confidence=confidence_threshold)
        detections = sv.Detections.from_inference(results).with_nms(
            class_agnostic=True, threshold=nms_threshold
        )

        self.sam.set_image(image, image_format="RGB")
        masks = [self.get_mask(xyxy) for xyxy in detections.xyxy]
        detections.mask = np.array(masks)

        return self.annotate_image(image, detections, categories)

    def get_mask(self, xyxy):
        mask, _, _ = self.sam.predict(box=xyxy, multimask_output=False)
        return mask.squeeze()

    def annotate_image(self, image, detections, categories):
        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        labels = [
            f"{categories[class_id]}: {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        for annotator in self.annotators.values():
            output_image = annotator.annotate(output_image, detections)
        return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


def download_test_image():
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
def create_app():
    segmenter = ImageSegmenter()
    download_test_image()
    return gr.Interface(
        fn=segmenter.detect,
        inputs=[
            gr.Image(type="numpy", label="input image"),
            gr.Text(info="you can input multiple words with comma (,)"),
            gr.Slider(minimum=0, maximum=1, value=0.3, step=0.01, interactive=True, label="Confidence Threshold"),
            gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, interactive=True, label="NMS Threshold"),
        ],
        outputs=gr.Image(type="numpy", label="output image"),
        allow_flagging="never",
        title="Fast Text to Segmentation with YOLO-World + EfficientViT SAM",
        description="...",
        examples=[
            [
                os.path.join(os.path.dirname(__file__), "bus.jpg"),
                "bus",
                0.05,
                0.5
            ],
            [
                os.path.join(os.path.dirname(__file__), "dogs.jpg"),
                "dog",
                0.05,
                0.5
            ],
            [
                os.path.join(os.path.dirname(__file__), "zidane.jpg"),
                "person",
                0.05,
                0.5
            ],
        ]
    )



if __name__ == '__main__':
    import os
    import urllib.request

    import os
    import subprocess

    # 设置下载链接和文件名
    # efficientvit_sam_url = "https://huggingface.co/han-cai/efficientvit-sam/resolve/main"
    # efficientvit_sam_url = "https://hf-mirror.com/mit-han-lab/efficientvit-sam/blob/main/"
    # efficientvit_sam_model = "xl1.pt"
    #
    # # 检查文件是否已经存在，如果不存在则下载
    # if not os.path.exists(efficientvit_sam_model):
    #     print("Downloading", efficientvit_sam_model, "...")
    #     subprocess.run(["wget", f"{efficientvit_sam_url}/{efficientvit_sam_model}"], check=True)
    app = create_app()
    app.launch(server_name="0.0.0.0")