import fnmatch
import os

import PIL
import cv2
import gradio as gr
from argparse import ArgumentParser

import numpy as np
import torch
from PIL.Image import Image
from mim import download
from mmengine.logging import print_log

from mmdet.apis import DetInferencer

ckpt_path = "./checkpoint"
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

mmdet_list = ['mask-rcnn_r50_fpn_albu-1x_coco', 'atss_r50_fpn_1x_coco', 'atss_r101_fpn_1x_coco',
              'autoassign_r50-caffe_fpn_1x_coco', 'boxinst_r50_fpn_ms-90k_coco', 'boxinst_r101_fpn_ms-90k_coco',
              'faster-rcnn_r50_fpn_carafe_1x_coco', 'mask-rcnn_r50_fpn_carafe_1x_coco',
              'cascade-rcnn_r50-caffe_fpn_1x_coco', 'cascade-rcnn_r50_fpn_1x_coco',
              'cascade-rcnn_r50_fpn_20e_coco', 'cascade-rcnn_r101-caffe_fpn_1x_coco',
              'cascade-rcnn_r101_fpn_1x_coco', 'cascade-rcnn_r101_fpn_20e_coco',
              'cascade-rcnn_x101-32x4d_fpn_1x_coco', 'cascade-rcnn_x101-32x4d_fpn_20e_coco',
              'cascade-rcnn_x101-64x4d_fpn_1x_coco', 'cascade-rcnn_x101_64x4d_fpn_20e_coco',
              'cascade-mask-rcnn_r50-caffe_fpn_1x_coco', 'cascade-mask-rcnn_r50_fpn_1x_coco',
              'cascade-mask-rcnn_r50_fpn_20e_coco', 'cascade-mask-rcnn_r101-caffe_fpn_1x_coco',
              'cascade-mask-rcnn_r101_fpn_1x_coco', 'cascade-mask-rcnn_r101_fpn_20e_coco',
              'cascade-mask-rcnn_x101-32x4d_fpn_1x_coco', 'cascade-mask-rcnn_x101-32x4d_fpn_20e_coco',
              'cascade-mask-rcnn_x101-64x4d_fpn_1x_coco', 'cascade-mask-rcnn_x101-64x4d_fpn_20e_coco',
              'cascade-mask-rcnn_r50-caffe_fpn_ms-3x_coco', 'cascade-mask-rcnn_r50_fpn_mstrain_3x_coco',
              'cascade-mask-rcnn_r101-caffe_fpn_ms-3x_coco', 'cascade-mask-rcnn_r101_fpn_ms-3x_coco',
              'cascade-mask-rcnn_x101-32x4d_fpn_ms-3x_coco', 'cascade-mask-rcnn_x101-32x8d_fpn_ms-3x_coco',
              'cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco', 'cascade-rpn_fast-rcnn_r50-caffe_fpn_1x_coco',
              'cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco', 'centernet_r18-dcnv2_8xb16-crop512-140e_coco',
              'centernet_r18_8xb16-crop512-140e_coco', 'centernet-update_r50-caffe_fpn_ms-1x_coco',
              'centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco',
              'condinst_r50_fpn_ms-poly-90k_coco_instance', 'conditional-detr_r50_8xb2-50e_coco.py',
              'cornernet_hourglass104_10xb5-crop511-210e-mstest_coco',
              'cornernet_hourglass104_8xb6-210e-mstest_coco', 'cornernet_hourglass104_32xb3-210e-mstest_coco',
              'mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco',
              'cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco',
              'cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco',
              'crowddet-rcnn_refine_r50_fpn_8xb2-30e_crowdhuman', 'crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman',
              'dab-detr_r50_8xb2-50e_coco.py', 'faster-rcnn_r50_fpn_dconv_c3-c5_1x_coco',
              'faster-rcnn_r50_fpn_dpool_1x_coco', 'faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco',
              'faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco', 'mask-rcnn_r50_fpn_dconv_c3-c5_1x_coco',
              'mask-rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco', 'mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco',
              'cascade-rcnn_r50_fpn_dconv_c3-c5_1x_coco', 'cascade-rcnn_r101-dconv-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_r50_fpn_dconv_c3-c5_1x_coco', 'cascade-mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco', 'faster-rcnn_r50_fpn_mdconv_c3-c5_1x_coco',
              'faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco', 'faster-rcnn_r50_fpn_mdpool_1x_coco',
              'mask-rcnn_r50_fpn_mdconv_c3-c5_1x_coco', 'mask-rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco',
              'ddod_r50_fpn_1x_coco', 'deformable-detr_r50_16xb2-50e_coco',
              'deformable-detr_refine_r50_16xb2-50e_coco', 'deformable-detr_refine_twostage_r50_16xb2-50e_coco',
              'cascade-rcnn_r50-rfp_1x_coco', 'cascade-rcnn_r50-sac_1x_coco',
              'detectors_cascade-rcnn_r50_1x_coco', 'htc_r50-rfp_1x_coco', 'htc_r50-sac_1x_coco',
              'detectors_htc-r50_1x_coco', 'detr_r50_8xb2-150e_coco', 'dino-4scale_r50_8xb2-12e_coco.py',
              'dino-4scale_r50_8xb2-24e_coco.py', 'dino-5scale_swin-l_8xb2-12e_coco.py',
              'dino-5scale_swin-l_8xb2-36e_coco.py', 'dh-faster-rcnn_r50_fpn_1x_coco',
              'atss_r50-caffe_fpn_dyhead_1x_coco', 'atss_r50_fpn_dyhead_1x_coco',
              'atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco', 'dynamic-rcnn_r50_fpn_1x_coco',
              'retinanet_effb3_fpn_8xb4-crop896-1x_coco', 'faster-rcnn_r50_fpn_attention_1111_1x_coco',
              'faster-rcnn_r50_fpn_attention_0010_1x_coco', 'faster-rcnn_r50_fpn_attention_1111_dcn_1x_coco',
              'faster-rcnn_r50_fpn_attention_0010_dcn_1x_coco', 'faster-rcnn_r50-caffe-c4_1x_coco',
              'faster-rcnn_r50-caffe-c4_mstrain_1x_coco', 'faster-rcnn_r50-caffe-dc5_1x_coco',
              'faster-rcnn_r50-caffe_fpn_1x_coco', 'faster-rcnn_r50_fpn_1x_coco',
              'faster-rcnn_r50_fpn_fp16_1x_coco', 'faster-rcnn_r50_fpn_2x_coco',
              'faster-rcnn_r101-caffe_fpn_1x_coco', 'faster-rcnn_r101_fpn_1x_coco',
              'faster-rcnn_r101_fpn_2x_coco', 'faster-rcnn_x101-32x4d_fpn_1x_coco',
              'faster-rcnn_x101-32x4d_fpn_2x_coco', 'faster-rcnn_x101-64x4d_fpn_1x_coco',
              'faster-rcnn_x101-64x4d_fpn_2x_coco', 'faster-rcnn_r50_fpn_iou_1x_coco',
              'faster-rcnn_r50_fpn_giou_1x_coco', 'faster-rcnn_r50_fpn_bounded_iou_1x_coco',
              'faster-rcnn_r50-caffe-dc5_mstrain_1x_coco', 'faster-rcnn_r50-caffe-dc5_mstrain_3x_coco',
              'faster-rcnn_r50-caffe_fpn_ms-2x_coco', 'faster-rcnn_r50-caffe_fpn_ms-3x_coco',
              'faster-rcnn_r50_fpn_mstrain_3x_coco', 'faster-rcnn_r101-caffe_fpn_ms-3x_coco',
              'faster-rcnn_r101_fpn_ms-3x_coco', 'faster-rcnn_x101-32x4d_fpn_ms-3x_coco',
              'faster-rcnn_x101-32x8d_fpn_ms-3x_coco', 'faster-rcnn_x101-64x4d_fpn_ms-3x_coco',
              'faster-rcnn_r50_fpn_tnr-pretrain_1x_coco', 'fcos_r50-caffe_fpn_gn-head_1x_coco',
              'fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco',
              'fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco',
              'fcos_r101-caffe_fpn_gn-head-1x_coco', 'fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco',
              'fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco', 'fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco',
              'fovea_r50_fpn_4xb4-1x_coco', 'fovea_r50_fpn_4xb4-2x_coco',
              'fovea_r50_fpn_gn-head-align_4xb4-2x_coco', 'fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco',
              'fovea_r101_fpn_4xb4-1x_coco', 'fovea_r101_fpn_4xb4-2x_coco',
              'fovea_r101_fpn_gn-head-align_4xb4-2x_coco',
              'fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco', 'faster-rcnn_r50_fpg_crop640-50e_coco',
              'faster-rcnn_r50_fpg-chn128_crop640-50e_coco', 'mask-rcnn_r50_fpg_crop640-50e_coco',
              'mask-rcnn_r50_fpg-chn128_crop640-50e_coco', 'retinanet_r50_fpg_crop640_50e_coco',
              'retinanet_r50_fpg-chn128_crop640_50e_coco', 'freeanchor_r50_fpn_1x_coco',
              'freeanchor_r101_fpn_1x_coco', 'freeanchor_x101-32x4d_fpn_1x_coco', 'fsaf_r50_fpn_1x_coco',
              'fsaf_r101_fpn_1x_coco', 'fsaf_x101-64x4d_fpn_1x_coco', 'mask-rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco',
              'mask-rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco', 'mask-rcnn_r101-gcb-r16-c3-c5_fpn_1x_coco',
              'mask-rcnn_r101-gcb-r4-c3-c5_fpn_1x_coco', 'mask-rcnn_r50_fpn_syncbn-backbone_1x_coco',
              'mask-rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco',
              'mask-rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco', 'mask-rcnn_r101-syncbn_fpn_1x_coco',
              'mask-rcnn_r101-syncbn-gcb-r16-c3-c5_fpn_1x_coco',
              'mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco', 'mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco',
              'mask-rcnn_x101-32x4d-syncbn-gcb-r16-c3-c5_fpn_1x_coco',
              'mask-rcnn_x101-32x4d-syncbn-gcb-r4-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-syncbn-r16-gcb-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-syncbn-r4-gcb-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r16-gcb-c3-c5_fpn_1x_coco',
              'cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r4-gcb-c3-c5_fpn_1x_coco', 'gfl_r50_fpn_1x_coco',
              'gfl_r50_fpn_ms-2x_coco', 'gfl_r101_fpn_ms-2x_coco', 'gfl_r101-dconv-c3-c5_fpn_ms-2x_coco',
              'gfl_x101-32x4d_fpn_ms-2x_coco', 'gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco',
              'retinanet_r50_fpn_ghm-1x_coco', 'retinanet_r101_fpn_ghm-1x_coco',
              'retinanet_x101-32x4d_fpn_ghm-1x_coco', 'retinanet_x101-64x4d_fpn_ghm-1x_coco',
              'mask-rcnn_r50_fpn_gn-all_2x_coco', 'mask-rcnn_r50_fpn_gn-all_3x_coco',
              'mask-rcnn_r101_fpn_gn-all_2x_coco', 'mask-rcnn_r101_fpn_gn-all_3x_coco',
              'mask-rcnn_r50_fpn_gn-all_contrib_2x_coco', 'mask-rcnn_r50_fpn_gn-all_contrib_3x_coco',
              'faster-rcnn_r50_fpn_gn_ws-all_1x_coco', 'faster-rcnn_r101_fpn_gn-ws-all_1x_coco',
              'faster-rcnn_x50-32x4d_fpn_gn-ws-all_1x_coco', 'faster-rcnn_x101-32x4d_fpn_gn-ws-all_1x_coco',
              'mask-rcnn_r50_fpn_gn_ws-all_2x_coco', 'mask-rcnn_r101_fpn_gn-ws-all_2x_coco',
              'mask-rcnn_x50-32x4d_fpn_gn-ws-all_2x_coco', 'mask-rcnn_x101-32x4d_fpn_gn-ws-all_2x_coco',
              'mask-rcnn_r50_fpn_gn_ws-all_20_23_24e_coco', 'mask-rcnn_r101_fpn_gn-ws-all_20-23-24e_coco',
              'mask-rcnn_x50-32x4d_fpn_gn-ws-all_20-23-24e_coco',
              'mask-rcnn_x101-32x4d_fpn_gn-ws-all_20-23-24e_coco', 'grid-rcnn_r50_fpn_gn-head_2x_coco',
              'grid-rcnn_r101_fpn_gn-head_2x_coco', 'grid-rcnn_x101-32x4d_fpn_gn-head_2x_coco',
              'grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco', 'faster-rcnn_r50_fpn_groie_1x_coco',
              'grid-rcnn_r50_fpn_gn-head-groie_1x_coco', 'mask-rcnn_r50_fpn_groie_1x_coco',
              'mask-rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco',
              'mask-rcnn_r101_fpn_syncbn-r4-gcb_c3-c5-groie_1x_coco', 'ga-rpn_r50-caffe_fpn_1x_coco',
              'ga-rpn_r101-caffe_fpn_1x_coco', 'ga-rpn_x101-32x4d_fpn_1x_coco', 'ga-rpn_x101-64x4d_fpn_1x_coco',
              'ga-faster-rcnn_r50-caffe_fpn_1x_coco', 'ga-faster-rcnn_r101-caffe_fpn_1x_coco',
              'ga-faster-rcnn_x101-32x4d_fpn_1x_coco', 'ga-faster-rcnn_x101-64x4d_fpn_1x_coco',
              'ga-retinanet_r50-caffe_fpn_1x_coco', 'ga-retinanet_r101-caffe_fpn_1x_coco',
              'ga-retinanet_x101-32x4d_fpn_1x_coco', 'ga-retinanet_x101-64x4d_fpn_1x_coco',
              'faster-rcnn_hrnetv2p-w18-1x_coco', 'faster-rcnn_hrnetv2p-w18-2x_coco',
              'faster-rcnn_hrnetv2p-w32-1x_coco', 'faster-rcnn_hrnetv2p-w32_2x_coco',
              'faster-rcnn_hrnetv2p-w40-1x_coco', 'faster-rcnn_hrnetv2p-w40_2x_coco',
              'mask-rcnn_hrnetv2p-w18-1x_coco', 'mask-rcnn_hrnetv2p-w18-2x_coco',
              'mask-rcnn_hrnetv2p-w32-1x_coco', 'mask-rcnn_hrnetv2p-w32-2x_coco',
              'mask-rcnn_hrnetv2p-w40_1x_coco', 'mask-rcnn_hrnetv2p-w40-2x_coco',
              'cascade-rcnn_hrnetv2p-w18-20e_coco', 'cascade-rcnn_hrnetv2p-w32-20e_coco',
              'cascade-rcnn_hrnetv2p-w40-20e_coco', 'cascade-mask-rcnn_hrnetv2p-w18_20e_coco',
              'cascade-mask-rcnn_hrnetv2p-w32_20e_coco', 'cascade-mask-rcnn_hrnetv2p-w40-20e_coco',
              'htc_hrnetv2p-w18_20e_coco', 'htc_hrnetv2p-w32_20e_coco', 'htc_hrnetv2p-w40_20e_coco',
              'fcos_hrnetv2p-w18-gn-head_4xb4-1x_coco', 'fcos_hrnetv2p-w18-gn-head_4xb4-2x_coco',
              'fcos_hrnetv2p-w32-gn-head_4xb4-1x_coco', 'fcos_hrnetv2p-w32-gn-head_4xb4-2x_coco',
              'fcos_hrnetv2p-w18-gn-head_ms-640-800-4xb4-2x_coco',
              'fcos_hrnetv2p-w32-gn-head_ms-640-800-4xb4-2x_coco',
              'fcos_hrnetv2p-w40-gn-head_ms-640-800-4xb4-2x_coco', 'htc_r50_fpn_1x_coco', 'htc_r50_fpn_20e_coco',
              'htc_r101_fpn_20e_coco', 'htc_x101-32x4d_fpn_16xb1-20e_coco', 'htc_x101-64x4d_fpn_16xb1-20e_coco',
              'htc_x101-64x4d-dconv-c3-c5_fpn_ms-400-1400-16xb1-20e_coco',
              'mask-rcnn_r50_fpn_instaboost_4x_coco', 'mask-rcnn_r101_fpn_instaboost-4x_coco',
              'mask-rcnn_x101-64x4d_fpn_instaboost-4x_coco', 'cascade-mask-rcnn_r50_fpn_instaboost_4x_coco',
              'lad_r101-paa-r50_fpn_2xb8_coco_1x', 'lad_r50-paa-r101_fpn_2xb8_coco_1x',
              'ld_r18-gflv1-r101_fpn_1x_coco', 'ld_r34-gflv1-r101_fpn_1x_coco', 'ld_r50-gflv1-r101_fpn_1x_coco',
              'ld_r101-gflv1-r101-dcn_fpn_2x_coco', 'libra-faster-rcnn_r50_fpn_1x_coco',
              'libra-faster-rcnn_r101_fpn_1x_coco', 'libra-faster-rcnn_x101-64x4d_fpn_1x_coco',
              'libra-retinanet_r50_fpn_1x_coco', 'mask-rcnn_r50_fpn_sample1e-3_ms-2x_lvis-v0.5',
              'mask-rcnn_r101_fpn_sample1e-3_ms-2x_lvis-v0.5',
              'mask-rcnn_x101-32x4d_fpn_sample1e-3_ms-2x_lvis-v0.5',
              'mask-rcnn_x101-64x4d_fpn_sample1e-3_ms-2x_lvis-v0.5',
              'mask-rcnn_r50_fpn_sample1e-3_ms-1x_lvis-v1', 'mask-rcnn_r101_fpn_sample1e-3_ms-1x_lvis-v1',
              'mask-rcnn_x101-32x4d_fpn_sample1e-3_ms-1x_lvis-v1',
              'mask-rcnn_x101-64x4d_fpn_sample1e-3_ms-1x_lvis-v1',
              'mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic', 'mask2former_r101_8xb2-lsj-50e_coco',
              'mask2former_r101_8xb2-lsj-50e_coco-panoptic', 'mask2former_r50_8xb2-lsj-50e_coco-panoptic',
              'mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic', 'mask2former_r50_8xb2-lsj-50e_coco',
              'mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic',
              'mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic',
              'mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic',
              'mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco', 'mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco',
              'mask-rcnn_r50-caffe_fpn_1x_coco', 'mask-rcnn_r50_fpn_1x_coco', 'mask-rcnn_r50_fpn_fp16_1x_coco',
              'mask-rcnn_r50_fpn_2x_coco', 'mask-rcnn_r101-caffe_fpn_1x_coco', 'mask-rcnn_r101_fpn_1x_coco',
              'mask-rcnn_r101_fpn_2x_coco', 'mask-rcnn_x101-32x4d_fpn_1x_coco',
              'mask-rcnn_x101-32x4d_fpn_2x_coco', 'mask-rcnn_x101-64x4d_fpn_1x_coco',
              'mask-rcnn_x101-64x4d_fpn_2x_coco', 'mask-rcnn_x101-32x8d_fpn_1x_coco',
              'mask-rcnn_r50-caffe_fpn_ms-poly-2x_coco', 'mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco',
              'mask-rcnn_r50_fpn_mstrain-poly_3x_coco', 'mask-rcnn_r101_fpn_ms-poly-3x_coco',
              'mask-rcnn_r101-caffe_fpn_ms-poly-3x_coco', 'mask-rcnn_x101-32x4d_fpn_ms-poly-3x_coco',
              'mask-rcnn_x101-32x8d_fpn_ms-poly-1x_coco', 'mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco',
              'mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco', 'maskformer_r50_ms-16xb1-75e_coco',
              'maskformer_swin-l-p4-w12_64xb1-ms-300e_coco', 'ms-rcnn_r50-caffe_fpn_1x_coco',
              'ms-rcnn_r50-caffe_fpn_2x_coco', 'ms-rcnn_r101-caffe_fpn_1x_coco',
              'ms-rcnn_r101-caffe_fpn_2x_coco', 'ms-rcnn_x101-32x4d_fpn_1x_coco',
              'ms-rcnn_x101-64x4d_fpn_1x_coco', 'ms-rcnn_x101-64x4d_fpn_2x_coco',
              'nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco',
              'nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco', 'retinanet_r50_fpn_crop640-50e_coco',
              'retinanet_r50_nasfpn_crop640-50e_coco', 'faster-rcnn_r50_fpn_32x2_1x_openimages',
              'retinanet_r50_fpn_32xb2-1x_openimages', 'ssd300_32xb8-36e_openimages',
              'faster-rcnn_r50_fpn_32x2_1x_openimages_challenge', 'faster-rcnn_r50_fpn_32x2_cas_1x_openimages',
              'faster-rcnn_r50_fpn_32x2_cas_1x_openimages_challenge', 'paa_r50_fpn_1x_coco',
              'paa_r50_fpn_1.5x_coco', 'paa_r50_fpn_2x_coco', 'paa_r50_fpn_mstrain_3x_coco',
              'paa_r101_fpn_1x_coco', 'paa_r101_fpn_2x_coco', 'paa_r101_fpn_mstrain_3x_coco',
              'faster-rcnn_r50_pafpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco',
              'panoptic_fpn_r50_fpn_mstrain_3x_coco', 'panoptic_fpn_r101_fpn_1x_coco',
              'panoptic_fpn_r101_fpn_mstrain_3x_coco', 'retinanet_pvt-t_fpn_1x_coco',
              'retinanet_pvt-s_fpn_1x_coco', 'retinanet_pvt-m_fpn_1x_coco', 'retinanet_pvtv2-b0_fpn_1x_coco',
              'retinanet_pvtv2-b1_fpn_1x_coco', 'retinanet_pvtv2-b2_fpn_1x_coco',
              'retinanet_pvtv2-b3_fpn_1x_coco', 'retinanet_pvtv2-b4_fpn_1x_coco',
              'retinanet_pvtv2-b5_fpn_1x_coco', 'pisa_faster_rcnn_r50_fpn_1x_coco',
              'pisa_faster_rcnn_x101_32x4d_fpn_1x_coco', 'pisa_mask_rcnn_r50_fpn_1x_coco',
              'pisa_retinanet_r50_fpn_1x_coco', 'pisa_retinanet_x101_32x4d_fpn_1x_coco', 'pisa_ssd300_coco',
              'pisa_ssd512_coco', 'point_rend_r50_caffe_fpn_mstrain_1x_coco',
              'point_rend_r50_caffe_fpn_mstrain_3x_coco', 'queryinst_r50_fpn_1x_coco',
              'queryinst_r50_fpn_ms-480-800-3x_coco', 'queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_coco',
              'queryinst_r101_fpn_ms-480-800-3x_coco',
              'queryinst_r101_fpn_300-proposals_crop-ms-480-800-3x_coco', 'mask-rcnn_regnetx-3.2GF_fpn_1x_coco',
              'mask-rcnn_regnetx-4GF_fpn_1x_coco', 'mask-rcnn_regnetx-6.4GF_fpn_1x_coco',
              'mask-rcnn_regnetx-8GF_fpn_1x_coco', 'mask-rcnn_regnetx-12GF_fpn_1x_coco',
              'mask-rcnn_regnetx-3.2GF-mdconv-c3-c5_fpn_1x_coco', 'faster-rcnn_regnetx-3.2GF_fpn_1x_coco',
              'faster-rcnn_regnetx-3.2GF_fpn_2x_coco', 'retinanet_regnetx-800MF_fpn_1x_coco',
              'retinanet_regnetx-1.6GF_fpn_1x_coco', 'retinanet_regnetx-3.2GF_fpn_1x_coco',
              'faster-rcnn_regnetx-400MF_fpn_ms-3x_coco', 'faster-rcnn_regnetx-800MF_fpn_ms-3x_coco',
              'faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco', 'faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco',
              'faster-rcnn_regnetx-4GF_fpn_ms-3x_coco', 'mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco',
              'mask-rcnn_regnetx-400MF_fpn_ms-poly-3x_coco', 'mask-rcnn_regnetx-800MF_fpn_ms-poly-3x_coco',
              'mask-rcnn_regnetx-1.6GF_fpn_ms-poly-3x_coco', 'mask-rcnn_regnetx-4GF_fpn_ms-poly-3x_coco',
              'cascade-mask-rcnn_regnetx-400MF_fpn_ms-3x_coco', 'cascade-mask-rcnn_regnetx-800MF_fpn_ms-3x_coco',
              'cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco', 'cascade-mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco',
              'cascade-mask-rcnn_regnetx-4GF_fpn_ms-3x_coco', 'reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco',
              'reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco', 'reppoints-moment_r50_fpn_1x_coco',
              'reppoints-moment_r50_fpn-gn_head-gn_1x_coco', 'reppoints-moment_r50_fpn-gn_head-gn_2x_coco',
              'reppoints-moment_r101_fpn-gn_head-gn_2x_coco',
              'reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco',
              'reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco', 'faster-rcnn_res2net-101_fpn_2x_coco',
              'mask-rcnn_res2net-101_fpn_2x_coco', 'cascade-rcnn_res2net-101_fpn_20e_coco',
              'cascade-mask-rcnn_res2net-101_fpn_20e_coco', 'htc_res2net-101_fpn_20e_coco',
              'faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco',
              'faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco',
              'mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco',
              'mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco',
              'cascade-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco',
              'cascade-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco',
              'cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco',
              'cascade-mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco',
              'faster-rcnn_r50_fpn_rsb-pretrain_1x_coco', 'cascade-mask-rcnn_r50_fpn_rsb-pretrain_1x_coco',
              'retinanet_r50-rsb-pre_fpn_1x_coco', 'mask-rcnn_r50_fpn_rsb-pretrain_1x_coco',
              'retinanet_r18_fpn_1x_coco', 'retinanet_r18_fpn_1xb8-1x_coco', 'retinanet_r50-caffe_fpn_1x_coco',
              'retinanet_r50_fpn_1x_coco', 'retinanet_r50_fpn_amp-1x_coco', 'retinanet_r50_fpn_2x_coco',
              'retinanet_r50_fpn_ms-640-800-3x_coco', 'retinanet_r101-caffe_fpn_1x_coco',
              'retinanet_r101-caffe_fpn_ms-3x_coco', 'retinanet_r101_fpn_1x_coco', 'retinanet_r101_fpn_2x_coco',
              'retinanet_r101_fpn_ms-640-800-3x_coco', 'retinanet_x101-32x4d_fpn_1x_coco',
              'retinanet_x101-32x4d_fpn_2x_coco', 'retinanet_x101-64x4d_fpn_1x_coco',
              'retinanet_x101-64x4d_fpn_2x_coco', 'retinanet_x101-64x4d_fpn_ms-640-800-3x_coco',
              'rpn_r50-caffe_fpn_1x_coco', 'rpn_r50_fpn_1x_coco', 'rpn_r50_fpn_2x_coco',
              'rpn_r101-caffe_fpn_1x_coco', 'rpn_x101-32x4d_fpn_1x_coco', 'rpn_x101-32x4d_fpn_2x_coco',
              'rpn_x101-64x4d_fpn_1x_coco', 'rpn_x101-64x4d_fpn_2x_coco', 'rtmdet_tiny_8xb32-300e_coco',
              'rtmdet_s_8xb32-300e_coco', 'rtmdet_m_8xb32-300e_coco', 'rtmdet_l_8xb32-300e_coco',
              'rtmdet_x_8xb32-300e_coco', 'rtmdet-ins_tiny_8xb32-300e_coco', 'rtmdet-ins_s_8xb32-300e_coco',
              'rtmdet-ins_m_8xb32-300e_coco', 'rtmdet-ins_l_8xb32-300e_coco', 'rtmdet-ins_x_8xb16-300e_coco',
              'sabl-faster-rcnn_r50_fpn_1x_coco', 'sabl-faster-rcnn_r101_fpn_1x_coco',
              'sabl-cascade-rcnn_r50_fpn_1x_coco', 'sabl-cascade-rcnn_r101_fpn_1x_coco',
              'sabl-retinanet_r50_fpn_1x_coco', 'sabl-retinanet_r50-gn_fpn_1x_coco',
              'sabl-retinanet_r101_fpn_1x_coco', 'sabl-retinanet_r101-gn_fpn_1x_coco',
              'sabl-retinanet_r101-gn_fpn_ms-640-800-2x_coco', 'sabl-retinanet_r101-gn_fpn_ms-480-960-2x_coco',
              'scnet_r50_fpn_1x_coco', 'scnet_r50_fpn_20e_coco', 'scnet_r101_fpn_20e_coco',
              'scnet_x101-64x4d_fpn_20e_coco', 'faster-rcnn_r50_fpn_gn-all_scratch_6x_coco',
              'mask-rcnn_r50_fpn_gn-all_scratch_6x_coco',
              'mask-rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1',
              'mask-rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1',
              'mask-rcnn_r101_fpn_seesaw-loss_random-ms-2x_lvis-v1',
              'mask-rcnn_r101_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1',
              'mask-rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1',
              'mask-rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1',
              'mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1',
              'mask-rcnn_r101_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1',
              'cascade-mask-rcnn_r101_fpn_seesaw-loss_random-ms-2x_lvis-v1',
              'cascade-mask-rcnn_r101_fpn_seesaw-loss-normed-mask_random-ms-2x_lvis-v1',
              'cascade-mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1',
              'cascade-mask-rcnn_r101_fpn_seesaw-loss-normed-mask_sample1e-3-ms-2x_lvis-v1',
              'mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco',
              'mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco',
              'mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco',
              'mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco',
              'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco.py',
              'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco.py',
              'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco.py',
              'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py', 'sparse-rcnn_r50_fpn_1x_coco',
              'sparse-rcnn_r50_fpn_ms-480-800-3x_coco',
              'sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco',
              'sparse-rcnn_r101_fpn_ms-480-800-3x_coco',
              'sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco', 'decoupled-solo_r50_fpn_1x_coco',
              'decoupled-solo_r50_fpn_3x_coco', 'decoupled-solo-light_r50_fpn_3x_coco', 'solo_r50_fpn_3x_coco',
              'solo_r50_fpn_1x_coco', 'solov2_r50_fpn_1x_coco', 'solov2_r50_fpn_ms-3x_coco',
              'solov2_r101-dcn_fpn_ms-3x_coco', 'solov2_x101-dcn_fpn_ms-3x_coco',
              'solov2-light_r18_fpn_ms-3x_coco', 'solov2-light_r50_fpn_ms-3x_coco', 'ssd300_coco', 'ssd512_coco',
              'ssdlite_mobilenetv2-scratch_8xb24-600e_coco',
              'mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco',
              'mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco', 'mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco',
              'mask-rcnn_swin-t-p4-w7_fpn_1x_coco', 'mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco',
              'tridentnet_r50-caffe_1x_coco', 'tridentnet_r50-caffe_ms-1x_coco',
              'tridentnet_r50-caffe_ms-3x_coco', 'tood_r101_fpn_ms-2x_coco', 'tood_x101-64x4d_fpn_ms-2x_coco',
              'tood_r101-dconv-c3-c5_fpn_ms-2x_coco', 'tood_r50_fpn_anchor-based_1x_coco',
              'tood_r50_fpn_1x_coco', 'tood_r50_fpn_ms-2x_coco', 'vfnet_r50_fpn_1x_coco',
              'vfnet_r50_fpn_ms-2x_coco', 'vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco', 'vfnet_r101_fpn_1x_coco',
              'vfnet_r101_fpn_ms-2x_coco', 'vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco',
              'vfnet_x101-32x4d-mdconv-c3-c5_fpn_ms-2x_coco', 'vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco',
              'yolact_r50_1x8_coco', 'yolact_r50_8x8_coco', 'yolact_r101_1x8_coco', 'yolov3_d53_320_273e_coco',
              'yolov3_d53_mstrain-416_273e_coco', 'yolov3_d53_mstrain-608_273e_coco',
              'yolov3_d53_fp16_mstrain-608_273e_coco', 'yolov3_mobilenetv2_8xb24-320-300e_coco',
              'yolov3_mobilenetv2_8xb24-ms-416-300e_coco', 'yolof_r50_c5_8x8_1x_coco', 'yolox_s_8x8_300e_coco',
              'yolox_l_8x8_300e_coco', 'yolox_x_8x8_300e_coco', 'yolox_tiny_8x8_300e_coco',
              'bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval',
              'bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17test',
              'bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot20train_test-mot20test',
              'strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval',
              'strongsort_yolox_x_8xb4-80e_crowdhuman-mot20train_test-mot20test',
              'ocsort_yolox_x_crowdhuman_mot17-private-half',
              'sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval',
              'deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval',
              'qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval',
              'mask2former_r50_8xb2-8e_youtubevis2021', 'mask2former_r101_8xb2-8e_youtubevis2021',
              'mask2former_swin-l-p4-w12-384-in21k_8xb2-8e_youtubevis2021.py',
              'masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019',
              'masktrack-rcnn_mask-rcnn_r101_fpn_8xb1-12e_youtubevis2019',
              'masktrack-rcnn_mask-rcnn_x101_fpn_8xb1-12e_youtubevis2019',
              'masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2021',
              'masktrack-rcnn_mask-rcnn_r101_fpn_8xb1-12e_youtubevis2021',
              'masktrack-rcnn_mask-rcnn_x101_fpn_8xb1-12e_youtubevis2021',
              'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365', 'glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365',
              'glip_atss_swin-t_c_fpn_dyhead_pretrain_obj365-goldg',
              'glip_atss_swin-t_fpn_dyhead_pretrain_obj365-goldg-cc3m-sub',
              'glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata']

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

def save_image(img, img_path):
    # Convert PIL image to OpenCV image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Save OpenCV image
    cv2.imwrite(img_path, img)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, help='Input image file or folder path.')
    gr.inputs.Dropdown(choices=[m for m in mmdet_list], label='Model',
                       default='rtmdet_tiny_8xb32-300e_coco'),
    parser.add_argument(
        '--model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
             'and alias defined in metafile. The model configuration '
             'file will try to read from .pth if the parameter is '
             'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument('--texts', help='text prompt', default=None)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
             'If so, the input text should be '
             '"cls_name1 . cls_name2 . cls_name3 ." format')

    call_args = vars(parser.parse_args())
    return call_args


def clear_folder(folder_path):
    import shutil
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print(f"Clear {folder_path} successfully.")


def download_cfg_checkpoint_model_name(model_name):
    clear_folder("./checkpoint")
    download(package='mmdet',
             configs=[model_name],
             dest_root='./checkpoint')


def detect_objects(img_path, model, weights, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis,
                   no_save_pred, print_result, palette, custom_entities):
    call_args = parse_args()
    call_args['model'] = model
    call_args['weights'] = weights
    call_args['inputs'] = img_path
    call_args['device'] = device
    call_args['out_dir'] = out_dir
    call_args['texts'] = texts
    call_args['pred_score_thr'] = float(pred_score_thr)
    call_args['batch_size'] = int(batch_size)
    call_args['show'] = show
    call_args['no_save_vis'] = no_save_vis
    call_args['no_save_pred'] = no_save_pred
    call_args['print_result'] = print_result
    call_args['palette'] = palette
    call_args['custom_entities'] = custom_entities

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main(inputs, model_name, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis, no_save_pred,
         print_result, palette, custom_entities):
    download_cfg_checkpoint_model_name(model_name)
    img_path = "input_img.jpg"
    save_image(inputs, img_path)
    # inputs.save("input_img.jpg")
    path = "./checkpoint"
    model = [f for f in os.listdir(path) if fnmatch.fnmatch(f, model_name + "*.py")][0]
    model = path + "/" + model

    weights = [f for f in os.listdir(path) if fnmatch.fnmatch(f, model_name + "*.pth")][0]
    weights = path + "/" + weights
    init_args, call_args = detect_objects(img_path, model, weights, out_dir, texts, device, pred_score_thr, batch_size,
                                          show, no_save_vis, no_save_pred,
                                          print_result, palette, custom_entities)

    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    inferencer = DetInferencer(**init_args)
    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')
    save_dir = './outputs/vis/'
    img_out = PIL.Image.open(os.path.join(save_dir, img_path))
    return img_out


if __name__ == '__main__':
    download_test_image()
    examples = [
        ['bus.jpg', 'rtmdet_tiny_8xb32-300e_coco', './outputs', '', "cpu"],
        ['dogs.jpg', 'mask-rcnn_r50_fpn_albu-1x_coco', './outputs', '', "cpu"],
        ['zidane.jpg', 'yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco', './outputs', '', "cpu"]
    ]

    title = "MMDetection detection web demo"
    description = "<div style='text-align:center'><img src='https://raw.githubusercontent.com/open-mmlab/mmdetection/main/resources/mmdet-logo.png'></div>" \
                  "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmdetection'>MMDetection</a> 是一个开源的物体检测工具箱，提供了丰富的检测模型和数据增强方式。" \
                  "OpenMMLab Detection Toolbox and Benchmark.</p>"
    article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmdetection'>MMDetection</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

    iface = gr.Interface(
        fn=main,
        inputs=[
            gr.inputs.Image(type="pil", label="input"),
            gr.inputs.Dropdown(choices=[m for m in mmdet_list], label='Model',
                               default='rtmdet_tiny_8xb32-300e_coco'),
            gr.inputs.Textbox(label="out_dir", default="./outputs/"),
            gr.inputs.Textbox(label="texts", default=''),
            gr.inputs.Textbox(label="device", default="cpu"),
            gr.inputs.Slider(label="pred_score_thr", minimum=0.0, maximum=1.0, step=0.1, default=0.3),
            gr.inputs.Number(label="batch_size", default=1),
            gr.inputs.Checkbox(label="show"),
            gr.inputs.Checkbox(label="no_save_vis"),
            gr.inputs.Checkbox(label="no_save_pred"),
            gr.inputs.Checkbox(label="print_result"),
            gr.inputs.Radio(label="palette", choices=["coco", "voc", "citys", "random", "none"]),
            gr.inputs.Checkbox(label="custom_entities")
        ],
        outputs=gr.outputs.Image(type="pil"),
        examples=examples,
        title=title,
        description=description, article=article, allow_flagging=False
    )

    iface.launch()
