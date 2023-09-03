import cv2
import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import SAM

# Load the model
model = SAM('mobile_sam.pt')

def predict(image: Image, bbox: str) -> np.ndarray:
    # Parse the bounding box string
    bbox_list = list(map(int, bbox.split(',')))

    # Predict a segment based on a box prompt
    result = model.predict(image, bboxes=bbox_list)
    plotted = result[0].plot()
    plotted = cv2.cvtColor(np.array(plotted), cv2.COLOR_BGR2RGB)
    return plotted

# Define the Gradio interface
image_input = gr.inputs.Image(type='pil')
text_input = gr.inputs.Textbox(lines=1, label="Bounding Box (x1, y1, x2, y2)")
image_output = gr.outputs.Image('pil')

iface = gr.Interface(fn=predict, inputs=[image_input, text_input], outputs=image_output)

# Launch the Gradio interface
iface.launch()