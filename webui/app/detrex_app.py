import os
os.system("git clone https://github.com/IDEA-Research/detrex.git")
os.system("python -m pip install git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2")
os.system("cd detrex && pip install -e .")
os.system("pip install fairscale")

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
    "conditional_detr/conditional_detr_r50_50ep":{
        "configs":"projects/conditional_detr/configs/conditional_detr_r50_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/conditional_detr_r50_50ep.pth"
    },
    "conditional_detr/conditional_detr_r50_50ep_(converted)":{
        "configs":"",
        "ckpts":""
    },
    "conditional_detr/conditional_detr_r101_50ep":{
        "configs":"projects/conditional_detr/configs/conditional_detr_r101_50ep.py",
        "ckpts":"https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/converted_conditional_detr_r101_50ep.pth"
    },
    "conditional_detr/conditional_detr_r101_dc5_50ep": {
        "configs": "projects/conditional_detr/configs/conditional_detr_r101_dc5_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_conditional_detr_r101_dc5.pth"
    },
    # DAB-DETR
    "dab_detr/dab_detr_r50_50ep": {
        "configs": "projects/dab_detr/configs/dab_detr_r50_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_r50_50ep.pth"
    },
    "dab_detr/dab_detr_r50_3patterns_50ep": {
        "configs": "projects/dab_detr/configs/dab_detr_r50_3patterns_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_dab_detr_r50_3patterns.pth"
    },
    "dab_detr/dab_detr_r50_dc5_50ep": {
        "configs": "projects/dab_detr/configs/dab_detr_r50_dc5_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_dab_detr_r50_dc5.pth"
    },
    "dab_detr/dab_detr_r50_dc5_3patterns_50ep": {
        "configs": "projects/dab_detr/configs/dab_detr_r50_3patterns_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_dab_detr_r50_dc5_3patterns.pth"
    },
    "dab_detr/dab_detr_r101_50ep": {
        "configs": "projects/dab_detr/configs/dab_detr_r101_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_r101_50ep.pth"
    },
    "dab_detr/dab_detr_r50_dc5_3patterns_50ep_(converted)": {
        "configs": "projects/dab_detr/configs/dab_detr_r50_dc5_3patterns_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_detr_r101_dc5.pth"
    },
    "dab_detr/dab_detr_swin_t_in1k_50ep": {
        "configs": "projects/dab_detr/configs/dab_detr_swin_t_in1k_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dab_detr_swin_t_in1k_50ep.pth"
    },
    "dab_detr/dab_deformable_detr_r50_50ep": {
        "configs": "projects/dab_detr/configs/dab_deformable_detr_r50_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dab_deformable_detr_r50_50ep_49AP.pth"
    },
    "dab_detr/dab_deformable_detr_r50_two_stage_50ep": {
        "configs": "projects/dab_detr/configs/dab_deformable_detr_r50_two_stage_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dab_deformable_detr_r50_two_stage_49_7AP.pth"
    },
    # DN-DETR
    "dn_detr/dn_detr_r50_50ep": {
        "configs": "projects/dn_detr/configs/dn_detr_r50_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.0/dn_detr_r50_50ep.pth"
    },
    "dn_detr/dn_detr_r50_dc5_50ep": {
        "configs": "projects/dn_detr/configs/dn_detr_r50_dc5_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_dn_detr_r50_dc5.pth"
    },
    # DINO
    "dino/dino-resnet/dino_r50_5scale_12ep": {
        "configs": "projects/dino/configs/dino-resnet/dino_r50_5scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_r50_5scale_12ep.pth"
    },
    "dino/dino-resnet/dino_r50_4scale_12ep_300dn": {
        "configs": "projects/dino/configs/dino-resnet/dino_r50_4scale_12ep_300dn.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_r50_4scale_12ep_300dn.pth"
    },
    "dino/dino-resnet/dino_r50_4scale_24ep": {
        "configs": "projects/dino/configs/dino-resnet/dino_r50_4scale_24ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r50_4scale_24ep.pth"
    },
    "dino/dino-resnet/dino_r101_4scale_12ep_": {
        "configs": "projects/dino/configs/dino-resnet/dino_r101_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_r101_4scale_12ep.pth"
    },
    # Pretrained DINO with Swin-Transformer Backbone
    "dino/dino-swin/dino_swin_tiny_224_4scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_small_224_4scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_small_224_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_base_384_4scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_base_384_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_base_384_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_large_224_4scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_large_224_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_224_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_large_384_4scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_large_384_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_large_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_large_384_5scale_12ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_large_384_5scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_swin_large_384_5scale_12ep.pth"
    },
    "dino/dino-swin/dino_swin_large_384_4scale_36ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_large_384_4scale_36ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_swin_large_384_4scale_36ep.pth"
    },
    "dino/dino-swin/dino_swin_large_384_5scale_36ep": {
        "configs": "projects/dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_swin_large_384_5scale_36ep.pth"
    },
    # Pretrained DINO with FocalNet Backbone
    "dino/dino-swin/dino_focalnet_large_lrf_384_4scale_12ep": {
        "configs": "projects/dino/configs/dino-focal/dino_focalnet_large_lrf_384_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_focal_large_lrf_384_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_focalnet_large_lrf_384_fl4_4scale_12ep": {
        "configs": "projects/dino/configs/dino-focal/dino_focalnet_large_lrf_384_fl4_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_focal_large_lrf_384_4scale_12ep.pth"
    },
    "dino/dino-swin/dino_focalnet_large_lrf_384_fl4_5scale_12ep": {
        "configs": "projects/dino/configs/dino-focal/dino_focalnet_large_lrf_384_fl4_5scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_focalnet_large_lrf_384_fl4_5scale_12ep.pth"
    },
    # Pretrained DINO with ViTDet Backbone
    "dino/dino-vitdet/dino_vitdet_base_4scale_12ep": {
        "configs": "projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_4scale_12ep.pth"
    },
    "dino/dino-vitdet/dino_vitdet_base_4scale_50ep": {
        "configs": "projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_base_4scale_50ep.pth"
    },
    "dino/dino-vitdet/dino_vitdet_large_4scale_12ep": {
        "configs": "projects/dino/configs/dino-vitdet/dino_vitdet_large_4scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_large_4scale_12ep.pth"
    },
    "dino/dino-vitdet/dino_vitdet_large_4scale_50ep": {
        "configs": "projects/dino/configs/dino-vitdet/dino_vitdet_large_4scale_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.1/dino_vitdet_large_4scale_50ep.pth"
    },
    # H-Deformable-DETR
    "h_deformable_detr/h_deformable_detr_r50_two_stage_12ep": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_r50_two_stage_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/h_deformable_detr_r50_two_stage_12ep_modified_train_net.pth"
    },
    "h_deformable_detr/h_deformable_detr_r50_two_stage_12ep(converted)": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_r50_two_stage_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_r50_two_stage_36ep": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_r50_two_stage_36ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_swin_tiny_two_stage_12ep": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_swin_tiny_two_stage_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_swin_tiny_two_stage_36ep": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_swin_tiny_two_stage_36ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_swin_large_two_stage_12ep": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_swin_large_two_stage_36ep": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_36ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_swin_large_two_stage_12ep_900queries": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_12ep_900queries.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth"
    },
    "h_deformable_detr/h_deformable_detr_swin_large_two_stage_36ep_900queries": {
        "configs": "projects/h_deformable_detr/configs/h_deformable_detr_swin_large_two_stage_36ep_900queries.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth"
    },
    # DETA
    "deta/improved_deformable_detr_baseline_50ep": {
        "configs": "projects/deta/configs/improved_deformable_detr_baseline_50ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_deta_improved_deformable_baseline.pth"
    },
    "deta/deta_r50_5scale_12ep_bs8": {
        "configs": "projects/deta/configs/deta_r50_5scale_12ep_bs8.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/deta_r50_5scale_12ep_bs8.pth"
    },
    "deta/deta_r50_5scale_12ep": {
        "configs": "projects/deta/configs/deta_r50_5scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/deta_r50_5scale_12ep_hacked_trainer.pth"
    },
    "deta/deta_r50_5scale_no_frozen_backbone": {
        "configs": "projects/deta/configs/deta_r50_5scale_no_frozen_backbone.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.4.0/deta_r50_5scale_12ep_no_freeze_backbone.pth"
    },
    "deta/deta_r50_5scale_12ep(converted)": {
        "configs": "projects/deta/configs/deta_r50_5scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_deta_r50_5scale_12ep.pth"
    },
    "deta/DETA-Swin-Large-finetune (converted)": {
        "configs": "projects/deta/configs/deta_r50_5scale_12ep.py",
        "ckpts": "https://github.com/IDEA-Research/detrex-storage/releases/download/v0.3.0/converted_deta_swin_o365_finetune.pth"
    },
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


def download_ckpts_and_image(ckpts):
    print("ckpts:", ckpts)
    torch.hub.download_url_to_file(ckpts, "dino_deitsmall16_pretrain.pth")

def run_detection(input_file, output_file, model_name, input_confidence, device):

    configs = detrex_model_list[model_name]["configs"]
    ckpts = detrex_model_list[model_name]["ckpts"]

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

def download_test_img():
    import shutil
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12658779/projects.zip',
        'projects.zip')
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/268517006-d8d4d3b3-964a-4f4d-8458-18c7eb75a4f2.jpg',
        '000000502136.jpg')
    shutil.unpack_archive('projects.zip', './', 'zip')

def detect_image(input_image, model_name, input_confidence, device):
    input_dir = "input.jpg"
    input_image.save(input_dir)
    output_image = "output.jpg"
    run_detection(input_dir, output_image, model_name, input_confidence, device)
    return output_image


if __name__ == '__main__':
    input_image = gr.inputs.Image(type='pil', label="Input Image")
    input_model_name = gr.inputs.Dropdown(list(detrex_model_list.keys()), label="Model Name", default="dab_detr/dab_detr_r50_50ep")
    input_confidence = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.25, label="Confidence Threshold")
    input_device = gr.inputs.Radio(["cpu", "cuda"], label="Device", default="cpu")
    output_image = gr.outputs.Image(type='pil', label="Output Image")
    download_test_img()
    examples = [["000000502136.jpg", "dab_detr/dab_detr_r50_50ep", 0.25, "cpu"]]
    title = "🦖detrex: Benchmarking Detection Transformers web demo"
    description = "<div align='center'><img src='https://raw.githubusercontent.com/IDEA-Research/detrex/main/assets/logo_2.png' width='250''/><div>" \
                  "<p style='text-align: center'><a href='https://github.com/IDEA-Research/detrex'>detrex</a> detrex detrex 是一个开源工具箱，提供最先进的基于 Transformer 的检测算法。它建立在Detectron2之上，其模块设计部分借鉴了MMDetection和DETR。非常感谢他们组织良好的代码。主分支适用于Pytorch 1.10+或更高版本（我们推荐Pytorch 1.12）。" \
                  "detrex is a research platform for DETR-based object detection, segmentation, pose estimation and other visual recognition tasks.</p>"
    article = "<p style='text-align: center'><a href='https://github.com/IDEA-Research/detrex'>detrex</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

    image_interface = gr.Interface(detect_image,
                                   inputs=[input_image, input_model_name, input_confidence, input_device],
                                   outputs=output_image,examples=examples,
                                   title=title, article=article, description=description)
    image_interface.launch()
