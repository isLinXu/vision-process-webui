import cv2
import gradio as gr
import numpy as np
from PIL.Image import Image
from ultralytics import SAM
import warnings

warnings.filterwarnings("ignore")

class SAMModel:
    def __init__(self):
        model_path = '../../weights/mobile_sam/mobile_sam.pt'
        self.model = SAM(model_path)

    def mobilesam_point_predict(self, image, x, y):
        result = self.model.predict(image, points=[x, y], labels=[1])
        plotted = result[0].plot()
        plotted = cv2.cvtColor(np.array(plotted), cv2.COLOR_BGR2RGB)
        return plotted

    def mobile_bbox_predict(self, image: Image, bbox: str) -> np.ndarray:
        # Parse the bounding box string
        bbox_list = list(map(int, bbox.split(',')))

        # Predict a segment based on a box prompt
        result = self.model.predict(image, bboxes=bbox_list)
        plotted = result[0].plot()
        plotted = cv2.cvtColor(np.array(plotted), cv2.COLOR_BGR2RGB)
        return plotted


    def launch(self):
        """Launches the Gradio interface."""
        # Create the UI
        with gr.Blocks() as app:
            # Header
            gr.Markdown("# SAM Model Demo")

            # Tabs
            with gr.Tabs():
                # Point-predict-button Tab
                with gr.TabItem("point-predict"):
                    with gr.Column():
                        inputs = [
                            gr.inputs.Image(type='pil', label='Input Image'),
                            gr.inputs.Number(default=900, label='X Coordinate'),
                            gr.inputs.Number(default=370, label='Y Coordinate'),
                        ]

                    output = gr.outputs.Image(type='pil', label='Output Image')
                    point_predict_button = gr.Button("inference")

                    # Run object detection on the input image when the button is clicked
                    point_predict_button.click(self.mobilesam_point_predict,
                                           inputs=inputs,
                                           outputs=output)

                # Bbox-predict-button Tab
                with gr.TabItem("bbox-predict"):
                    image_input = gr.inputs.Image(type='pil')
                    text_input = gr.inputs.Textbox(lines=1, label="Bounding Box (x1, y1, x2, y2)", default="439, 437, 524, 709")
                    image_output = gr.outputs.Image('pil')
                    inputs = [image_input, text_input]
                    output = image_output
                    point_predict_button = gr.Button("inference")

                    # Run object detection on the input image when the button is clicked
                    point_predict_button.click(self.mobile_bbox_predict,
                                               inputs=inputs,
                                               outputs=output)

                app.launch(share=True)


if __name__ == '__main__':
    web_ui = SAMModel()
    web_ui.launch()
