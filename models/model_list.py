
from list.yolo_model_list import yolonas_model_list, yolov3_model_list, yolov5_model_list, yolov8_model_list, \
    yolox_model_list, ppyoloe_model_list

from list.mmpretrain_model_list import mmpretrain_model_list
from list.mmyolo_model_list import mmyolo_configs_list
from list.mmdet_model_list import mmdet_model_list
from list.mmseg_model_list import mmsegmentation_model_list
from list.mmaction2_model_list import mmaction2_models_list
from list.mmrotate_model_list import mmrorate_model_list
from list.mmocr_model_list import textdet_model_list, textrec_model_list, textkie_model_list
from list.mmpose_model_list import mmpose_model_list
from list.mmagic_model_list import super_resolution_model_list, image_to_image_model_list, text_to_image_model_list

from list.damo_face_list import damo_face_list
from list.yolo_model_list import damo_model_list

from list.timm_cls_list import timm_model_list
from list.torchvision_cls_list import torchvision_cls_list
from list.torchvision_det_list import torchvision_det_list
from list.rtdetr_model_list import rtdetr_model_list
from list.detrex_model_list import detrex_model_list
from list.detectron2_model_list import detectron2_model_list


if __name__ == '__main__':
    print("yolov5_model_list:", len(yolov5_model_list))
    print("yolonas_model_list:", len(yolonas_model_list))
    print("yolov3_model_list:", len(yolov3_model_list))
    print("yolonas_model_list:", len(yolonas_model_list))
    print("ppyoloe_model_list:", len(ppyoloe_model_list))
    print("yolov8_model_list:", len(yolov8_model_list))
    print("yolox_model_list:", len(yolox_model_list))

    print("mmpretrain_model_list:", len(mmpretrain_model_list))
    print("mmyolo_configs_list:", len(mmyolo_configs_list))
    print("mmdet_model_list:", len(mmdet_model_list))
    print("mmsegmentation_model_list:", len(mmsegmentation_model_list))
    print("mmaction2_models_list:", len(mmaction2_models_list))
    print("mmrorate_model_list:", len(mmrorate_model_list))
    print("mmocr:", len(textdet_model_list) + len(textrec_model_list) + len(textkie_model_list))
    print("mmpose_model_list:", len(mmpose_model_list))
    print("mmagic_model_list:", len(super_resolution_model_list) + len(image_to_image_model_list) + len(text_to_image_model_list))

    print("damo_face_list:", len(damo_face_list))
    print("damo_yolo_list:", len(damo_model_list))

    print("timm_model_list:", len(timm_model_list))
    print("torchvision_cls_list:", len(torchvision_cls_list))
    print("torchvision_det_list:", len(torchvision_det_list))
    print("rtdetr_model_list:", len(rtdetr_model_list))
    print("detrex_model_list:", len(detrex_model_list))
    print("detectron2_model_list:", len(detectron2_model_list))
