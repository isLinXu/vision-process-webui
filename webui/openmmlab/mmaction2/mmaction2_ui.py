import argparse
import fnmatch
import os.path
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

import torch
from mmengine import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer
from mim import download

import warnings
warnings.filterwarnings("ignore")

import gradio as gr

mmaction2_models_list = [
    'slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb',
    'slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb',
    'slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb',
    'slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb',
    'slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb',
    'slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb',
    'slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb',
    'slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb',
    'slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb',
    'slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb',
    'vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb',
    'vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb',
    'c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb',
    'c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb',
    'c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb',
    'c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb',
    'c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb',
    'ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb',
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb',
    'ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb',
    'ipcsn_r152_32x2x1-180e_kinetics400-rgb',
    'ircsn_r152_32x2x1-180e_kinetics400-rgb',
    'ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb',
    'ipcsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb',
    'ircsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb',
    'i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb',
    'mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb_infer',
    'mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb',
    'mvit-base-p244_32x3x1_kinetics400-rgb', 'mvit-large-p244_40x3x1_kinetics400-rgb',
    'mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb_infer',
    'mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb',
    'mvit-base-p244_u32_sthv2-rgb', 'mvit-large-p244_u40_sthv2-rgb',
    'mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb',
    'slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb',
    'r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb',
    'r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb',
    'slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb',
    'slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb',
    'slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb',
    'slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb',
    'slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb',
    'slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb',
    'slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb',
    'slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb',
    'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb',
    'slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb',
    'slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb',
    'slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb',
    'slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb',
    'slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb',
    'slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb',
    'swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb',
    'swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb',
    'swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb',
    'tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb',
    'tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb',
    'tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb',
    'timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb',
    'timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb',
    'timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb',
    'tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb',
    'tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb',
    'tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb',
    'tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb',
    'tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb',
    'tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb',
    'trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb',
    'trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb',
    'tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb',
    'tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb',
    'tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb',
    'tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb',
    'uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb',
    'uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb',
    'uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb',
    'uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics400-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics400-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics400-rgb',
    'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics400-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics600-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics600-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics600-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics600-rgb',
    'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb',
    'uniformerv2-base-p16-res224_clip-pre_8xb32-u8_kinetics700-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics700-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics700-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics700-rgb',
    'uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics700-rgb',
    'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics700-rgb',
    'uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb',
    'uniformerv2-large-p14-res224_clip-pre_u8_kinetics710-rgb',
    'uniformerv2-large-p14-res336_clip-pre_u8_kinetics710-rgb',
    'uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_16xb32-u8_mitv1-rgb',
    'uniformerv2-large-p16-res224_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb',
    'uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb',
    'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400',
    'vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400',
    'vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400',
    'vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400',
    'x3d_s_13x6x1_facebook-kinetics400-rgb',
    'x3d_m_16x5x1_facebook-kinetics400-rgb',
    'tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature',
    'bmn_2xb8-400x100-9e_activitynet-feature',
    'bsn_400x100_1xb16_20e_activitynet_feature (cuhk_mean_100)',
    'clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb',
    '2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d',
    '2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d',
    '2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d',
    '2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d',
    '2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'slowonly_r50_8xb16-u48-240e_gym-keypoint',
    'slowonly_r50_8xb16-u48-240e_gym-limb', 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint',
    'slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb',
    'slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint',
    'slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint',
    'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d',
    'stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d',
    'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d',
    'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d',
    'stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d'
]

labelmap_list = [
    'kinetics_label_map_k400.txt', 'kinetics_label_map_k600.txt', 'kinetics_label_map_k700.txt',
    'kinetics_label_map_k710.txt', 'diving48_label_map.txt', 'gym_label_map.txt',
    'hmdb51_label_map.txt', 'jester_label_map.txt', 'mit_label_map.txt',
    'mmit_label_map.txt', 'multisports_label_map.txt', 'skeleton_label_map_gym99.txt',
    'skeleton_label_map_ntu60.txt', 'sthv1_label_map.txt', 'sthv2_label_map.txt', 'ucf101_label_map.txt',
]



def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument('--video', help='video file/url or rawframes directory')
    parser.add_argument('--label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
             'generate file')
    parser.add_argument(
        '--font-scale',
        default=None,
        type=float,
        help='font scale of the text in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the text in output video')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
             'video as input. If either dimension is set to -1, the frames are '
             'resized by keeping the existing aspect ratio')
    parser.add_argument('--out-filename', default=None, help='output filename')
    args = parser.parse_args()
    return args


def get_output(
        video_path: str,
        out_filename: str,
        data_sample: str,
        labels: list,
        fps: int = 30,
        font_scale: Optional[str] = None,
        font_color: str = 'white',
        target_resolution: Optional[Tuple[int]] = None,
) -> None:
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path.
        out_filename (str): Output filename for the generated file.
        datasample (str): Predicted label of the generated file.
        labels (list): Label list of current dataset.
        fps (int): Number of picture frames to read per second. Defaults to 30.
        font_scale (float): Font scale of the text. Defaults to None.
        font_color (str): Font color of the text. Defaults to ``white``.
        target_resolution (Tuple[int], optional): Set to
            (desired_width desired_height) to have resized frames. If
            either dimension is None, the frames are resized by keeping
            the existing aspect ratio. Defaults to None.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    # init visualizer
    out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
    visualizer = ActionVisualizer()
    visualizer.dataset_meta = dict(classes=labels)

    text_cfg = {'colors': font_color}
    if font_scale is not None:
        text_cfg.update({'font_sizes': font_scale})

    visualizer.add_datasample(
        out_filename,
        video_path,
        data_sample,
        draw_pred=True,
        draw_gt=False,
        text_cfg=text_cfg,
        fps=fps,
        out_type=out_type,
        out_path=osp.join('demo', out_filename),
        target_resolution=target_resolution)


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
    download(package='mmaction2',
             configs=[model_name],
             dest_root='./checkpoint')


def download_test_video():
    # Images
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/267197615-0e372587-9f42-428a-8f3b-e4e6f17e8b1a.mp4',
        'demo.mp4')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/267197620-56ee9562-ba3a-4ac4-977a-6df1cd693c39.mp4',
        'zelda.mp4')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/267197784-b8bff32a-6655-4777-a3f4-49070d480a76.mp4',
        'test_video_structuralize.mp4')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/267197798-9f88e0b9-1889-494a-a886-2e1e9ed43327.mp4',
        'shaowei.mp4')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/267197804-953056d5-1351-4c5c-8459-f4e8f6815836.mp4',
        'demo_skeleton.mp4')
    torch.hub.download_url_to_file(
        'https://user-images.githubusercontent.com/59380685/267197812-b4be4451-b694-4717-b8cf-545e36e506c1.mp4',
        'cxk.mp4')


def download_label_map_txt():
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579936/ucf101_label_map.txt',
        'ucf101_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579940/gym_label_map.txt',
        'gym_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579943/diving48_label_map.txt',
        'diving48_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579947/hmdb51_label_map.txt',
        'hmdb51_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579949/jester_label_map.txt',
        'jester_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579951/kinetics_label_map_k400.txt',
        'kinetics_label_map_k400.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579952/kinetics_label_map_k600.txt',
        'kinetics_label_map_k600.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579953/kinetics_label_map_k700.txt',
        'kinetics_label_map_k700.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579954/kinetics_label_map_k710.txt',
        'kinetics_label_map_k710.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579955/mit_label_map.txt',
        'mit_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579957/mmit_label_map.txt',
        'mmit_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579960/multisports_label_map.txt',
        'multisports_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579961/skeleton_label_map_ntu60.txt',
        'mmit_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579960/multisports_label_map.txt',
        'multisports_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579961/skeleton_label_map_ntu60.txt',
        'skeleton_label_map_ntu60.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579962/skeleton_label_map_gym99.txt',
        'skeleton_label_map_gym99.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579965/sthv1_label_map.txt',
        'sthv1_label_map.txt')
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/12579967/sthv2_label_map.txt',
        'sthv2_label_map.txt')


def mmaction_inference(video, mmaction2_models, device, label, out_filename):
    args = parse_args()
    path = "./checkpoint"
    if not os.path.exists(path):
        os.makedirs(path)
    download_cfg_checkpoint_model_name(mmaction2_models)
    config = [f for f in os.listdir(path) if fnmatch.fnmatch(f, "*.py")][0]
    config = path + "/" + config

    checkpoint = [f for f in os.listdir(path) if fnmatch.fnmatch(f, "*.pth")][0]
    checkpoint = path + "/" + checkpoint

    # args setting
    args.config = config
    args.checkpoint = checkpoint
    args.video = video
    args.device = device
    args.label = label
    args.out_filename = out_filename

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    pred_result = inference_recognizer(model, args.video)

    pred_scores = pred_result.pred_scores.item.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    labels = open(args.label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in top5_label]

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

    if args.out_filename is not None:

        if args.target_resolution is not None:
            if args.target_resolution[0] == -1:
                assert isinstance(args.target_resolution[1], int)
                assert args.target_resolution[1] > 0
            if args.target_resolution[1] == -1:
                assert isinstance(args.target_resolution[0], int)
                assert args.target_resolution[0] > 0
            args.target_resolution = tuple(args.target_resolution)

        get_output(
            args.video,
            args.out_filename,
            pred_result,
            labels,
            fps=args.fps,
            font_scale=args.font_scale,
            font_color=args.font_color,
            target_resolution=args.target_resolution)
        save_dir_path = "demo/" + args.out_filename
        if os.path.exists(save_dir_path):
            print(f'File saved as {save_dir_path}')
            return save_dir_path
        else:
            base_name = os.path.basename(args.video)
            print(f'File saved as {base_name}')
            return base_name


if __name__ == '__main__':
    print("Downloading test video and model...")
    download_test_video()
    print("Downloading label map txt...")
    download_label_map_txt()

    input_video = gr.Video(type='mp4', label="Original video")
    mmaction2_models = gr.inputs.Dropdown(label="MMAction2 models", choices=[x for x in mmaction2_models_list],
                                          default='tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb')
    device = gr.inputs.Radio(label="Device", choices=["cpu", "cuda:0"], default="cpu")
    # label = gr.inputs.Textbox(label="Label file", default="label_map/kinetics/label_map_k400.txt")
    label = gr.inputs.Dropdown(label="Label file", choices=[x for x in labelmap_list], default='kinetics_label_map_k400.txt')
    out_filename = gr.inputs.Textbox(label="Output filename", default="demo_dst.mp4")
    output_video = gr.Video(label="Output video")

    examples = [['zelda.mp4', 'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb', "cpu",
                 'kinetics_label_map_k400.txt', "demo_dst.mp4"],
                ['shaowei.mp4', 'slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb', "cpu",
                 'kinetics_label_map_k400.txt', "demo_dst.mp4"],
                ['baoguo.mp4', 'slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb', "cpu",
                 'kinetics_label_map_k400.txt', "demo_dst.mp4"],
                ['cxk.mp4', 'slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb', "cpu",
                 'kinetics_label_map_k400.txt', "demo_dst.mp4"]
                ]

    title = "MMAction2 web demo"
    description = "<div align='center'><img src='https://raw.githubusercontent.com/open-mmlab/mmaction2/main/resources/mmaction2_logo.png' width='450''/><div>" \
                  "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmaction2'>MMAction2</a> MMAction2 是一款基于 PyTorch 开发的行为识别开源工具包，是 open-mmlab 项目的一个子项目。" \
                  "OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark.</p>"
    article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmaction2'>MMAction2</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/isLinXu'>gradio build by gatilin</a></a></p>"

    # gradio demo
    iface = gr.Interface(fn=mmaction_inference,
                         inputs=[input_video, mmaction2_models, device, label, out_filename],
                         outputs=output_video,examples=examples,
                         title=title, description=description, article=article)
    iface.launch()
