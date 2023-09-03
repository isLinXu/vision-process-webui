import cv2
import gradio as gr
import numpy as np
from ultralytics import SAM
import warnings

warnings.filterwarnings("ignore")

class SAMModel:
    def __init__(self, model_path):
        self.model = SAM(model_path)

    def predict(self, image, x, y):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        result = self.model.predict(image, points=[x, y], labels=[1])
        result = result[0]
        plotted = result[0].plot()
        return plotted

def sam_demo(model_path: str, title: str, description: str) -> None:
    sam_model = SAMModel(model_path)

    inputs = [
        gr.inputs.Image(type='pil', label='Input Image'),
        gr.inputs.Number(default=900, label='X Coordinate'),
        gr.inputs.Number(default=370, label='Y Coordinate'),
    ]

    output = gr.outputs.Image(type='pil', label='Output Image')

    app = gr.Interface(fn=sam_model.predict, inputs=inputs, outputs=output,
                       title=title,
                       description=description)

    app.launch()
if __name__ == '__main__':
    sam_demo('mobile_sam.pt', 'SAM Model Demo', 'Use the SAM model to predict a segment based on a point prompt.')