import cv2
import gradio as gr
import numpy as np
from ultralytics import RTDETR
import warnings

warnings.filterwarnings("ignore")


class RT_DETR_WebUI:
    def __init__(self):
        pass

    # Define a function for model inference
    def predict(self, image, conf, iou, line_width, device, model_type, model_path):
        # choose model type
        if model_type == "rtdetr-l":
            self.model = RTDETR('rtdetr-l.pt')
        elif model_type == "rtdetr-x":
            self.model = RTDETR('rtdetr-x.pt')
        else:
            self.model = RTDETR(model_path)
        results = self.model(image)
        res = results[0]
        save_dir = res.save_dir
        path = res.path
        from PIL import Image
        dst = Image.open(save_dir + '/' + path)
        dst = cv2.cvtColor(np.array(dst), cv2.COLOR_RGB2BGR)
        return dst

if __name__ == '__main__':
    # Instantiate RT_DETR_WebUI class
    detector = RT_DETR_WebUI()

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
                gr.inputs.Radio(["rtdetr-l", "rtdetr-x"],
                                label="Model Type", default="rtdetr-l"),
                gr.inputs.Textbox(default="rtdetr-l.pt", label="Model Path")],
        outputs="image",
        title="RT_DETR Object Detector",
        description="Detect objects in an image using RT_DETR model.",
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
