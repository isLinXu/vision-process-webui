# """Fast text to segmentation with yolo-world and efficient-vit sam."""
import os
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
             "/Users/gatilin/PycharmProjects/vision-process-webui/images/demo.jpg"
            ],
            [
                os.path.join(os.path.dirname(__file__), "examples/livingroom.jpg"),
                "table, lamp, dog, sofa, plant, clock, carpet, frame on the wall",
                0.05,
                0.5
            ],
            [
                os.path.join(os.path.dirname(__file__), "examples/cat_and_dogs.jpg"),
                "cat, dog",
                0.2,
                0.5
            ],
        ],
    )


if __name__ == '__main__':
    app = create_app()
    app.launch(server_name="0.0.0.0")