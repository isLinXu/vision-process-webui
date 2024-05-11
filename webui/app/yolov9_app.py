import gradio as gr
import torch
# import spaces
import wget
import os

os.system("pip install yolov9pip==0.0.4")


def download_models(model_type):
    if model_type == "yolov9-c":
        url = 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt'
    elif model_type == "yolov9-e":
        url = 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt'
    elif model_type == "gelan-c":
        url = 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt'
    elif model_type == "gelan-e":
        url = 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt'
    else:
        raise ValueError("Invalid model type. Choose from 'yolov9-c', 'yolov9-e', 'gelan-e', or 'gelan-c'.")

    filename = wget.download(url)
    return filename


def yolov9_inference(img_path, model_type, image_size, conf_threshold, iou_threshold):
    import yolov9

    # Load the model
    model_path = download_models(model_type)
    model = yolov9.load(model_path, device="cpu")

    # Set model parameters
    model.conf = conf_threshold
    model.iou = iou_threshold

    # Perform inference
    results = model(img_path, size=image_size)

    # Optionally, show detection bounding boxes on image
    output = results.render()

    return output[0]


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


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Image")
                model_type = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov9-c",
                        "yolov9-e.pt",
                        "gelan-c.pt",
                        "gelan-e.pt",
                    ],
                    value="yolov9-c",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.4,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                )
                yolov9_infer = gr.Button(value="Inference")

            with gr.Column():
                output_numpy = gr.Image(type="numpy", label="Output")

        yolov9_infer.click(
            fn=yolov9_inference,
            inputs=[
                img_path,
                model_type,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
        )

        gr.Examples(
            examples=[
                [
                    "zidane.jpg", "yolov9-c", 640, 0.4, 0.5,
                ],
                [
                    "bus.jpg", "yolov9-c", 640, 0.4, 0.5,
                ],
                [
                    "dogs.jpg", "yolov9-c", 640, 0.4, 0.5,
                ],
            ],
            fn=yolov9_inference,
            inputs=[
                img_path,
                model_type,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
            cache_examples=True,
        )




if __name__ == '__main__':
    download_test_image()
    gradio_app = gr.Blocks()
    with gradio_app:
        gr.HTML(
            """
            <h1 style='text-align: center'>
            YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
            </h1>
            """)
        with gr.Row():
            with gr.Column():
                app()
    gradio_app.launch(debug=True)
