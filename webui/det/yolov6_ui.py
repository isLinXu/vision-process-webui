
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