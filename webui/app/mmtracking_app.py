import os
# os.system("pip install 'mmcv-full>=1.3.17,<=1.7.0'")
os.system("pip install 'mmcv-full>=1.3.17,<=1.7.0'")
os.system("pip install mmdet==2.25.1")
os.system("git clone https://github.com/open-mmlab/mmtracking.git")
os.system("pip install -r mmtracking/requirements.txt")
os.system("pip install -v -e mmtracking/")
os.system("pip install 'mmtrack'")
import os
import os.path as osp
import gradio as gr
import tempfile
from argparse import ArgumentParser

import mmcv

from mmtrack.apis import inference_mot, init_model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args
def track_mot(input, config, output, device, score_thr):
    args = parse_args()
    args.input = input
    args.config = config
    args.output = output
    args.device = device
    args.score_thr = score_thr
    args.show = False
    args.backend = 'cv2'

    # assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True
    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
    #
    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)
    #
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(model, img, frame_id=i)
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        model.show_result(
            img,
            result,
            score_thr=args.score_thr,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()
    # print("output:", out_dir)

    # return output
    # print("output:", out_dir)
    save_dir = 'mot.mp4'
    return save_dir

if __name__ == '__main__':
    # main()
    input_video = gr.Video(type="mp4", label="Input Video")
    config = gr.inputs.Textbox(default="configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py")
    output = gr.inputs.Textbox(default="mot.mp4", label="Output Video")
    device = gr.inputs.Radio(choices=["cpu", "cuda"], label="Device used for inference", default="cpu")
    score_thr = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.3, label="The threshold of score to filter bboxes.")
    output_video = gr.Video(type="mp4", label="Output Image")

    title = "MMTracking web demo"
    description = "<div align='center'><img src='https://raw.githubusercontent.com/open-mmlab/mmtracking/master/resources/mmtrack-logo.png' width='450''/><div>" \
                  "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmtracking'>MMTracking</a> MMTracking是一款基于PyTorch的视频目标感知开源工具箱，是OpenMMLab项目的一部分。" \
                  "OpenMMLab Video Perception Toolbox. It supports Video Object Detection (VID), Multiple Object Tracking (MOT), Single Object Tracking (SOT), Video Instance Segmentation (VIS) with a unified framework..</p>"
    article = "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmtracking'>MMTracking</a></p>" \
              "<p style='text-align: center'><a href='https://github.com/open-mmlab/mmtracking'>gradio build by gatilin</a></a></p>"

    # Create Gradio interface
    iface = gr.Interface(
        fn=track_mot,
        inputs=[
            input_video, config, output, device, score_thr
        ],
        # outputs="playable_video",
        outputs=output_video,
        title=title, description=description, article=article,
    )

    # Launch Gradio interface
    iface.launch()