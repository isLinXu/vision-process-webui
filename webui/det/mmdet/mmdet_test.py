
import gradio as gr
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer

def parse_args(inputs, model, weights, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis, no_save_pred, print_result, palette, custom_entities):
    call_args = {
        'inputs': inputs,
        'model': model,
        'weights': weights,
        'out_dir': out_dir,
        'texts': texts,
        'device': device,
        'pred_score_thr': pred_score_thr,
        'batch_size': batch_size,
        'show': show,
        'no_save_vis': no_save_vis,
        'no_save_pred': no_save_pred,
        'print_result': print_result,
        'palette': palette,
        'custom_entities': custom_entities
    }

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

def main(inputs, model, weights, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis, no_save_pred, print_result, palette, custom_entities):
    init_args, call_args = parse_args(inputs, model, weights, out_dir, texts, device, pred_score_thr, batch_size, show, no_save_vis, no_save_pred, print_result, palette, custom_entities)
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    inferencer = DetInferencer(**init_args)
    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')



iface = gr.Interface(
    parse_args,
    inputs=[
        gr.inputs.Textbox(label="inputs"),
        gr.inputs.Textbox(label="model"),
        gr.inputs.Textbox(label="weights"),
        gr.inputs.Textbox(label="out_dir"),
        gr.inputs.Textbox(label="texts"),
        gr.inputs.Textbox(label="device", default="cpu"),
        gr.inputs.Slider(label="pred_score_thr", minimum=0.0, maximum=1.0, step=0.1),
        gr.inputs.Number(label="batch_size", default=1),
        gr.inputs.Checkbox(label="show"),
        gr.inputs.Checkbox(label="no_save_vis"),
        gr.inputs.Checkbox(label="no_save_pred"),
        gr.inputs.Checkbox(label="print_result"),
        gr.inputs.Radio(label="palette", choices=["coco", "voc", "citys", "random", "none"]),
        gr.inputs.Checkbox(label="custom_entities")
    ],
    outputs=[
        gr.outputs.Textbox(label="init_args"),
        gr.outputs.Textbox(label="call_args")
    ]
)

iface.launch()