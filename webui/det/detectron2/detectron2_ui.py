import PIL.Image
import gradio as gr
import torch
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

import warnings
warnings.filterwarnings("ignore")

class VisualizationDemo:
    def __init__(self, cfg, device, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            cfg.defrost()
            cfg.MODEL.DEVICE = device

            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


detectron2_model_list = {
    # Cityscapes
    "Cityscapes/mask_rcnn_R_50_FPN":{
        "config_file": "configs/Cityscapes/mask_rcnn_R_50_FPN.yaml",
        "ckpts": "detectron2://Cityscapes/"
    },
    # COCO-Detection
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x":{
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x":{
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x":{
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x": {
        "config_file": "configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "ckpts": "detectron2://COCO-InstanceSegmentation/"
    },
    # COCO-Detection
    "COCO-Detection/mask_rcnn_X_101_32x8d_FPN_3x": {
        "config_file": "configs/COCO-Detection/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "ckpts": "detectron2://COCO-Detection/"
    },
}



def dtectron2_instance_inference(image, input_model_name, confidence_threshold, device):
    cfg = get_cfg()
    config_file = detectron2_model_list[input_model_name]["config_file"]
    ckpts = detectron2_model_list[input_model_name]["ckpts"]
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = ckpts
    cfg.MODEL.DEVICE = "cpu"
    cfg.output = "output_img.jpg"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    visualization_demo = VisualizationDemo(cfg, device=device)
    if image:
        intput_path = "intput_img.jpg"
        image.save(intput_path)
        image = read_image(intput_path, format="BGR")
        predictions, vis_output = visualization_demo.run_on_image(image)
        output_image = PIL.Image.fromarray(vis_output.get_image())
        # print("predictions: ", predictions)
        return output_image

def download_test_img():
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/268517006-d8d4d3b3-964a-4f4d-8458-18c7eb75a4f2.jpg',
        '000000502136.jpg')


if __name__ == '__main__':
    input_image = gr.inputs.Image(type='pil', label='Input Image')
    input_model_name = gr.inputs.Dropdown(list(detectron2_model_list.keys()), label="Model Name", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x")
    input_prediction_threshold = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.25, label="Confidence Threshold")
    input_device = gr.inputs.Dropdown(["cpu", "cuda"], label="Devices", default="cpu")
    output_image = gr.outputs.Image(type='pil', label='Output Image')
    output_predictions = gr.outputs.Textbox(type='text', label='Output Predictions')

    title = "Detectron2 web demo"
    description = "<div align='center'><img src='https://raw.githubusercontent.com/facebookresearch/detectron2/8c4a333ceb8df05348759443d0206302485890e0/.github/Detectron2-Logo-Horz.svg' width='450''/><div>" \
                  "<p style='text-align: center'><a href='https://github.com/facebookresearch/detectron2'>Detectron2</a> Detectron2 是 Facebook AI Research 的下一代库，提供最先进的检测和分割算法。它是Detectron 和maskrcnn-benchmark的后继者 。它支持 Facebook 中的许多计算机视觉研究项目和生产应用。" \
                  "Detectron2 is a platform for object detection, segmentation and other visual recognition tasks..</p>"
    article = "<p style='text-align: center'><a href='https://github.com/facebookresearch/detectron2'>Detectron2</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/facebookresearch/detectron2'>gradio build by gatilin</a></a></p>"
    download_test_img()

    examples = [["000000502136.jpg", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x", 0.25, "cpu"]]
    gr.Interface(fn=dtectron2_instance_inference,
                 inputs=[input_image, input_model_name, input_prediction_threshold, input_device],
                 outputs=output_image,examples=examples,
                 title=title, description=description, article=article).launch()
