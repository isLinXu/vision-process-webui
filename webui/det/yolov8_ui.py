import gradio as gr
from ultralytics import YOLO

import warnings

warnings.filterwarnings("ignore")

class YOLOv8WebUI:
    def __init__(self):
        pass

    def format_options(self, checkbox):
        """Formats the option list into an object."""
        option = {}
        for check in checkbox:
            option[check] = True
        return option

    def inference(self, type, input, checkbox, conf, iou, device, max_det, line_width, cpu):
        """Runs object detection on the input image."""
        # Load models
        if type == "detect":
            model = YOLO("../../weights/yolov8/yolov8n.pt")
        elif type == "seg":
            model = YOLO('../../weights/yolov8/yolov8n-seg.pt')
        elif type == "cls":
            model = YOLO('../../weights/yolov8/yolov8n-cls.pt')
        elif type == "pose":
            model = YOLO("../../weights/yolov8/yolov8n-pose.pt")

        # Set device to CPU if specified
        if cpu:
            device = "cpu"

        # Set line width to None if not specified
        if not line_width:
            line_width = None

        # Format the options
        option = self.format_options(checkbox)

        # Run object detection and plot the result
        res = model(input, conf=conf, iou=iou, device=device, max_det=max_det, line_width=line_width, **option)
        plotted = res[0].plot()

        return plotted

    def train(self, type: str, data_path: str, weights_path: str, epochs: int, batch_size: int, lr: float, device: str,
              img_size: int, resume: bool):
        """Trains the YOLOv8 model."""
        # Load model
        if type in self.model_files:
            model = YOLO(self.model_files[type])
        else:
            raise ValueError(f"Invalid task type: {type}")

        # Train model
        model.train(data_path=data_path, weights_path=weights_path, epochs=epochs, batch_size=batch_size, lr=lr,
                    device=device, img_size=img_size, resume=resume)

    def launch(self):
        """Launches the Gradio interface."""
        # Create the UI
        with gr.Blocks() as app:
            # Header
            gr.Markdown("# YOLOv8-WebUI")

            # Tabs
            with gr.Tabs():
                # Inference tab
                with gr.TabItem("inference"):
                    with gr.Row():
                        input = gr.Image()
                        output = gr.Image()

                    type = gr.Radio(["detect", "seg", "cls", "pose"], value="detect", label="Tasks")

                    # Options
                    conf = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.01, interactive=True, label="conf")
                    iou = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, interactive=True, label="iou")
                    checkbox = gr.CheckboxGroup(
                        ["half", "show", "save", "save_txt", "save_conf", "save_crop", "hide_labels", "hide_conf", "vid_stride",
                         "visualize", "augment", "agnostic_nms", "retina_masks", "boxes"], label="Options", value=["boxes"])

                    device = gr.Number(value=0, label="device", interactive=True, default="cpu",precision=0)
                    cpu = gr.Checkbox(label="cpu", interactive=True)
                    max_det = gr.Number(value=300, label="max_det", interactive=True, precision=0)
                    line_width = gr.Number(value=0, label="line_width", interactive=True, precision=0)

                    inference_button = gr.Button("inference")



            # Run object detection on the input image when the button is clicked
            inference_button.click(self.inference, inputs=[type, input, checkbox, conf, iou, device, max_det, line_width, cpu],
                                   outputs=output)

        app.launch(share=True)

if __name__ == '__main__':
    web_ui = YOLOv8WebUI()
    web_ui.launch()