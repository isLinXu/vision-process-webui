import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import warnings
import cv2
import torch
import tqdm
import gradio as gr

from demo.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import warnings

warnings.filterwarnings("ignore")

detrex_model_list = {
    # DETR
    "detr/detr_r50_300ep": {
        "configs": "projects/detr/configs/detr_r50_300ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/converted_detr_r50_500ep.pth"
    },
    "detr/detr_r50_dc5_300ep": {
        "configs": "projects/detr/configs/detr_r50_dc5_300ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_detr_r50_dc5.pth"
    },
    "detr/detr_r101_300ep.py": {
        "configs": "projects/detr/configs/detr_r101_300ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/converted_detr_r101_500ep.pth"
    },
    "detr/detr_r101_dc5_300ep.py": {
        "configs": "projects/detr/configs/detr_r101_dc5_300ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_detr_r101_dc5.pth"
    },
    # Deformable-DETR
    "deformable_detr/deformable_detr_r50_with_box_refinement_50ep": {
        "configs": "projects/deformable_detr/configs/deformable_detr_r50_with_box_refinement_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_with_box_refinement_50ep_new.pth"
    },
    "deformable_detr/deformable_detr_r50_two_stage_50ep": {
        "configs": "projects/deformable_detr/configs/deformable_detr_r50_two_stage_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/deformable_detr_r50_two_stage_50ep_new.pth"
    },
    # Anchor-DETR
    "anchor_detr/anchor_detr_r50_50ep":{
        "configs":"projects/anchor_detr/configs/anchor_detr_r50_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/anchor_detr_r50_50ep.pth"
    },
    "anchor_detr/anchor_detr_r50_50ep_(converted)":{
        "configs":"projects/anchor_detr/configs/anchor_detr_r50_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r50_50ep.pth"
    },
    "anchor_detr/anchor_detr_r50_dc5_50ep":{
        "configs":"projects/anchor_detr/configs/anchor_detr_r50_dc5_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r50_dc5_50ep.pth"
    },
    "anchor_detr/anchor_detr_r101_50ep":{
        "configs":"projects/anchor_detr/configs/anchor_detr_r101_dc5_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r101_dc5_50ep.pth"
    },
    "anchor_detr/anchor_detr_r101_dc5_50ep":{
        "configs":"projects/anchor_detr/configs/anchor_detr_r101_dc5_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_anchor_detr_r101_50ep.pth"
    },
    # Conditional-DETR


    # DAB-DETR


    # DN-DETR


    # DINO


    # Pretrained DINO with Swin-Transformer Backbone


    # Pretrained DINO with FocalNet Backbone


    # Pretrained DINO with ViTDet Backbone



    # H-Deformable-DETR


    # DETA

}


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="detrex demo for visualizing customized inputs")
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


# def download_ckpts_and_image():
#     torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", "dino_deitsmall16_pretrain.pth")


def run_detection(input_file, output_file, configs, ckpts, input_confidence, device):
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args([
        "--config-file", configs,
        "--input", input_file,
        "--output", output_file,
        "--confidence-threshold", str(input_confidence),
        "--opts", "train.init_checkpoint=" + ckpts
    ])
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)
    cfg.model.device = device
    cfg.train.device = device
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    demo = VisualizationDemo(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)


def detect_image(input_image, configs, ckpts, input_confidence, device):
    input_dir = "input.jpg"
    input_image.save(input_dir)
    output_image = "output.jpg"
    run_detection(input_dir, output_image, configs, ckpts, input_confidence, device)
    return output_image


if __name__ == '__main__':
    input_image = gr.inputs.Image(type='pil', label="Input Image")
    input_configs = gr.inputs.Textbox(label="Config File", default="projects/dab_detr/configs/dab_detr_r50_50ep.py")
    input_ckpts = gr.inputs.Textbox(label="Checkpoint File", default="dab_detr_r50_50ep.pth")
    input_confidence = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.25, label="Confidence Threshold")
    input_device = gr.inputs.Radio(["cpu", "cuda"], label="Device", default="cpu")
    output_image = gr.outputs.Image(type='pil', label="Output Image")

    image_interface = gr.Interface(detect_image,
                                   inputs=[input_image, input_configs, input_ckpts, input_confidence, input_device],
                                   outputs=output_image,
                                   title="Detrex Image Detection")
    image_interface.launch()
