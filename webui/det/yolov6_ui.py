# import gradio as gr
# import torch
# from PIL import Image
# import subprocess
# import os
# import PIL
# from pathlib import Path
# import uuid
#
# # Images
# torch.hub.download_url_to_file('https://miro.medium.com/max/1400/1*EYFejGUjvjPcc4PZTwoufw.jpeg', '1*EYFejGUjvjPcc4PZTwoufw.jpeg')
# torch.hub.download_url_to_file('https://production-media.paperswithcode.com/tasks/ezgif-frame-001_OZzxdny.jpg', 'ezgif-frame-001_OZzxdny.jpg')
# torch.hub.download_url_to_file('https://favtutor.com/resources/images/uploads/Social_Distancing_Covid_19__1.jpg', 'Social_Distancing_Covid_19__1.jpg')
# torch.hub.download_url_to_file('https://nkcf.org/wp-content/uploads/2017/11/people.jpg', 'people.jpg')
#
# def yolo(im):
#   file_name = str(uuid.uuid4())
#   im.save(f'{file_name}.jpg')
#   os.system(f"python /Users/gatilin/PycharmProjects/detection-webui/detection/yolov6/tools/tools/infer.py --weights yolov6s.pt --source {str(file_name)}.jpg --project ''")
#   img = PIL.Image.open(f"exp/{file_name}.jpg")
#   os.remove(f"exp/{file_name}.jpg")
#   os.remove(f'{file_name}.jpg')
#   return img
#
# inputs = gr.inputs.Image(type='pil', label="Original Image")
# outputs = gr.outputs.Image(type="pil", label="Output Image")
#
# title = "YOLOv6 - Demo"
# description = "YOLOv6 is a single-stage object detection framework dedicated to industrial applications, with hardware-friendly efficient design and high performance. Here is a quick Gradio Demo for testing YOLOv6s model. More details from  <a href='https://github.com/meituan/YOLOv6'>https://github.com/meituan/YOLOv6</a> "
# article = "<p>YOLOv6-nano achieves 35.0 mAP on COCO val2017 dataset with 1242 FPS on T4 using TensorRT FP16 for bs32 inference, and YOLOv6-s achieves 43.1 mAP on COCO val2017 dataset with 520 FPS on T4 using TensorRT FP16 for bs32 inference. More information at <a href='https://github.com/meituan/YOLOv6'>https://github.com/meituan/YOLOv6</a></p>"
#
# examples = [['1*EYFejGUjvjPcc4PZTwoufw.jpeg'], ['ezgif-frame-001_OZzxdny.jpg'], ['Social_Distancing_Covid_19__1.jpg'], ['people.jpg']]
#
# gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled = True, enable_queue=True).launch(inline=False, share=False, debug=False)

# import gradio as gr
#
# from detection.yolov6.tools.infer import run
#
#
# def yolov6_infer(weights, source, webcam, webcam_addr, yaml, img_size, conf_thres, iou_thres, max_det, device, save_txt, not_save_img, save_dir, view_img, classes, agnostic_nms, project, name, hide_labels, hide_conf, half):
#     run(weights, source, webcam, webcam_addr, yaml, img_size, conf_thres, iou_thres, max_det, device, save_txt, not_save_img, save_dir, view_img, classes, agnostic_nms, project, name, hide_labels, hide_conf, half)
#
# input_dict = {
#     "weights": gr.inputs.Textbox(default="weights/yolov6s.pt"),
#     "source": gr.inputs.Textbox(default="data/images"),
#     "webcam": gr.inputs.Checkbox(default=False),
#     "webcam_addr": gr.inputs.Textbox(default="0"),
#     "yaml": gr.inputs.Textbox(default="data/coco.yaml"),
#     "img_size": gr.inputs.Number(default=640),
#     "conf_thres": gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.4),
#     "iou_thres": gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.45),
#     "max_det": gr.inputs.Number(default=1000),
#     "device": gr.inputs.Textbox(default="0"),
#     "save_txt": gr.inputs.Checkbox(default=False),
#     "not_save_img": gr.inputs.Checkbox(default=False),
#     "save_dir": gr.inputs.Textbox(default=None),
#     "view_img": gr.inputs.Checkbox(default=True),
#     "classes": gr.inputs.Textbox(default=None),
#     "agnostic_nms": gr.inputs.Checkbox(default=False),
#     "project": gr.inputs.Textbox(default="runs/inference"),
#     "name": gr.inputs.Textbox(default="exp"),
#     "hide_labels": gr.inputs.Checkbox(default=False),
#     "hide_conf": gr.inputs.Checkbox(default=False),
#     "half": gr.inputs.Checkbox(default=False)
# }
#
# output_dict = {}
#
# interface = gr.Interface(fn=yolov6_infer, inputs=input_dict, outputs=output_dict, title="YOLOv6 PyTorch Inference",
#                          description="A web app for YOLOv6 PyTorch Inference. For more information, please refer to the README.md file.")
#
# interface.launch()

# import os
# import PIL
# import gradio as gr
#
#
# def demo_UI(img):
#     print('img', img)
#     input_length = len(os.listdir("/content/inputs"))
#     PIL.Image.fromarray(img).save(f'/content/inputs/imgs_input_{input_length + 1}.jpg')
#     os.system(f'bash run_yolov6.sh /content/inputs/imgs_input_{input_length + 1}.jpg /content/ouputs')
#     return PIL.Image.open(f'/content/ouputs/imgs_input_{input_length + 1}.jpg')
#
# demo = gr.Interface(
#     demo_UI,
#     inputs=gr.inputs.Image(),
#     outputs="image",
#     examples=[
#       ["/content/tokyo.jpg"],
#       ["/content/conan.jpg"]
#   ]
# )
#
# demo.launch(
#     debug=True,
#     share = True
# )

# import gradio as gr
# import subprocess
# import os
#
# def execute_command(file, weights, source):
#     # 设置环境变量
#     os.environ["PATH"] += os.pathsep + "/usr/local/bin"
#
#     # 执行命令
#     command = f"python tools/infer.py --weights {weights} --source {file.name}"
#     result = subprocess.run(command.split(), stdout=subprocess.PIPE)
#
#     # 输出结果
#     with open("output.jpg", "wb") as f:
#         f.write(result.stdout)
#     return "output.jpg"
#
# iface = gr.Interface(
#     fn=execute_command,
#     inputs=["file", "text", "text"],
#     outputs="file",
#     title="YOLOv6 推理演示",
#     description="输入一个权重文件和一个图像或视频文件，使用 YOLOv6 进行目标检测",
#     examples=[
#         ["img.jpg", "yolov6s.pt", "img.jpg"],
#         ["video.mp4", "yolov6s.pt", "video.mp4"],
#     ]
# )
#
# iface.launch()

import gradio as gr
import subprocess


def run_inference(weights, source):
    # command_1 = ["cd", ".."]
    # result = subprocess.run(command_1, capture_output=True, text=True)
    # command_2 = ["cd", "detection/yolov6"]
    # result = subprocess.run(command_2, capture_output=True, text=True)
    # 运行命令行命令
    command = ["python", "tools/infer.py", "--weights", weights, "--source", source]
    result = subprocess.run(command, capture_output=True, text=True)

    # 返回输出结果
    return result.stdout


# 创建 Gradio 接口
iface = gr.Interface(
    fn=run_inference,
    inputs=["text", "text"],
    outputs="text",
    title="YOLOv6 目标检测",
    description="使用 YOLOv6 进行目标检测。输入权重文件和源文件路径。",
    article="https://github.com/ultralytics/yolov6",
)

# 运行 Gradio 接口
iface.launch()