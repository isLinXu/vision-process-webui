import os
# os.system("pip install 'mmcv>=2.0.0'")
# os.system("pip install 'mmengine'")
# os.system("pip install 'mmagic'")
# os.system("pip install albumentations")
# os.system("pip install av")
# os.system("pip install accelerate")

import gradio as gr
from mmagic.apis import MMagicInferencer
import warnings

warnings.filterwarnings("ignore")

mmagic_model_list = [
    'aot_gan', 'basicvsr', 'basicvsr_pp',
    'biggan', 'cain', 'controlnet',
    'controlnet_animation',
    'cyclegan', 'dcgan', 'deblurganv2',
    'deepfillv1', 'deepfillv2',
    'dic', 'diffusers_pipeline',
    'dim', 'disco_diffusion',
    'draggan', 'dreambooth''edsr', 'edvr',
    'eg3d', 'esrgan', 'fastcomposer',
    'flavr', 'gca', 'ggan', 'glean',
    'global_local', 'guided_diffusion',
    'iconvsr', 'indexnet',
    'inst_colorization',
    'liif', 'lsgan', 'nafnet',
    'partial_conv', 'pggan', 'pix2pix',
    'positional_encoding_in_gans',
    'rdn', 'real_basicvsr',
    'real_esrgan', 'restormer',
    'sagan', 'singan', 'sngan_proj',
    'srcnn', 'srgan_resnet',
    'stable_diffusion',
    'styleganv1',
    'styleganv2',
    'styleganv3',
    'swinir', 'tdan',
    'textual_inversion', 'tof',
    'ttsr', 'vico', 'wgan-gp'
]

app_list = ['text_to_image', 'image_to_image', '3d_aware_generation',
            'image_super_resolution', 'image_inpainting',
            'image_matting', 'image_restoration', 'image_colorization']


def infer_image(text_prompts, image, app, model_name):
    if app == 'text_to_image':
        sd_inferencer = MMagicInferencer(model_name=model_name)
        result = sd_inferencer.infer(text=text_prompts, image=image)
    elif app == 'image_to_image':
        save_dir = 'input_img.jpg'
        result_out_dir = 'output_img.jpg'
        image.save(save_dir)
        # Create a MMagicInferencer instance and infer
        editor = MMagicInferencer('pix2pix')
        result = editor.infer(img=save_dir, result_out_dir=result_out_dir)
    return result


input_components = [
    gr.inputs.Textbox(label="Text Prompts", default="A panda is having dinner at KFC"),
    gr.inputs.Image(type='pil', label="Input Image"),
    gr.inputs.Radio(choices=app_list, label="MMagic Model", default="text_to_image"),
    gr.inputs.Dropdown(choices=mmagic_model_list, label="MMagic Model", default="stable_diffusion")
]

output_components = gr.outputs.Image(type='pil', label="Output Image")

gr.Interface(fn=infer_image, inputs=input_components, outputs=output_components).launch()
